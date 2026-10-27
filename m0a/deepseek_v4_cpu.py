"""DeepSeek-V4 attention component reference on CPU, using the mounted checkpoint.

Not a language model forward pass: caller supplies hidden states. No MoE, mHC,
other layers, language-model head or generated tokens. INT8 weights are decoded;
optional activation/index INT8 emulation is not a bit-exact Ascend kernel.
Equations checked against official DeepSeek inference/model.py and installed
vLLM-Ascend source; see docs/issue6-deepseek-cpu.md for provenance and scope.
"""
from __future__ import annotations
import os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


def digest(x):
    return hashlib.sha256(x.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class Checkpoint:
    def __init__(self, root):
        self.root = Path(root)
        self.config = json.loads((self.root/'config.json').read_text())
        self.index = json.loads((self.root/'quant_model_weights.safetensors.index.json').read_text())['weight_map']
        self.receipt = {}

    def tensor(self, name):
        with safe_open(str(self.root/self.index[name]), framework='pt', device='cpu') as f:
            x = f.get_tensor(name)
        self.receipt[name] = dict(shard=self.index[name], shape=list(x.shape), dtype=str(x.dtype),
                                  tensor_sha256=digest(x), bytes=x.numel()*x.element_size())
        return x

    def matrix(self, name):
        x = self.tensor(name+'.weight')
        quantized = x.dtype == torch.int8
        if quantized:
            scale = self.tensor(name+'.weight_scale').float()
            offset = self.tensor(name+'.weight_offset').float()
            if torch.count_nonzero(offset):
                raise ValueError('Unsupported nonzero quantization offset')
            if scale.shape != (x.shape[0], 1):
                raise ValueError('Unsupported quantization scale shape')
            x = x.float()*scale
        return x.float(), quantized


def rms(x, weight=None, eps=1e-6):
    y = x.float()*torch.rsqrt(x.float().square().mean(-1, keepdim=True)+eps)
    return y if weight is None else y*weight


def int8_reference(x):
    """Symmetric per-vector round/clamp/dequantize; rounding may differ from NPU."""
    scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-12)/127
    return torch.round(x/scale).clamp(-127, 127)*scale


def hadamard(x):
    d = x.shape[-1]
    if d <= 0 or d & (d-1):
        raise ValueError('Hadamard dimension must be a power of two')
    y = x.clone()
    width = 1
    while width < d:
        view = y.reshape(*y.shape[:-1], -1, 2, width)
        a, b = view[..., 0, :].clone(), view[..., 1, :].clone()
        y = torch.stack((a+b, a-b), dim=-2).reshape_as(y)
        width *= 2
    return y/math.sqrt(d)


def inverse_frequencies(config, compressed):
    d = config['qk_rope_head_dim']
    base = config['compress_rope_theta'] if compressed else config['rope_theta']
    spec = config['rope_scaling']
    freq = base**(-torch.arange(0, d, 2, dtype=torch.float32)/d)
    original = spec['original_max_position_embeddings']
    def correction(rotations):
        return d*math.log(original/(rotations*2*math.pi))/(2*math.log(base))
    low = max(0, math.floor(correction(spec['beta_fast'])))
    high = min(d-1, math.ceil(correction(spec['beta_slow'])))
    if low == high:
        high += .001
    ramp = ((torch.arange(d//2)-low)/(high-low)).clamp(0, 1)
    return freq*(1-ramp) + freq/spec['factor']*ramp


def rope(x, positions, inv_freq, *, inverse=False):
    """Interleaved RoPE on the last rotary dimensions, with leading token axis."""
    rd = inv_freq.numel()*2
    angles = positions.float().reshape(-1, 1)*inv_freq.reshape(1, -1)
    if inverse:
        angles = -angles
    shape = [len(positions)] + [1]*(x.ndim-2) + [rd//2]
    cosine, sine = angles.cos().reshape(shape), angles.sin().reshape(shape)
    y = x.clone()
    even, odd = x[..., -rd::2], x[..., -rd+1::2]
    y[..., -rd::2] = even*cosine-odd*sine
    y[..., -rd+1::2] = even*sine+odd*cosine
    return y


def pool_projected(kv, gate, ape, ratio):
    """Batch gated pooling: C4 previous first half + current second half; C128 current block."""
    count = len(kv)//ratio
    d = kv.shape[-1]//(2 if ratio == 4 else 1)
    if not count:
        return kv.new_empty((0, d))
    values = kv[:count*ratio].reshape(count, ratio, -1)
    logits = gate[:count*ratio].reshape(count, ratio, -1)+ape
    if ratio == 4:
        first_v = torch.zeros_like(values[..., :d])
        first_g = torch.full_like(logits[..., :d], -torch.inf)
        first_v[1:] = values[:-1, :, :d]
        first_g[1:] = logits[:-1, :, :d]
        values = torch.cat((first_v, values[..., d:]), dim=1)
        logits = torch.cat((first_g, logits[..., d:]), dim=1)
    return (values*torch.softmax(logits, dim=1)).sum(dim=1)


def pool_incremental(kv, gate, ape, ratio):
    """Independent token-by-token compressor reference, including partial blocks."""
    d = kv.shape[-1]//(2 if ratio == 4 else 1)
    ring_v, ring_g, result = [], [], []
    for p in range(len(kv)):
        ring_v.append(kv[p])
        ring_g.append(gate[p]+ape[p % ratio])
        if (p+1) % ratio:
            continue
        if ratio == 4:
            # Explicitly enumerate the contributing token/half pairs.
            pairs = [(t, 0 if t < p+1-ratio else d) for t in range(max(0, p+1-2*ratio), p+1)]
        else:
            pairs = [(t, 0) for t in range(p+1-ratio, p+1)]
        vals = torch.stack([ring_v[t][offset:offset+d] for t, offset in pairs])
        logs = torch.stack([ring_g[t][offset:offset+d] for t, offset in pairs])
        # Manual stable normalization, independent of the batched softmax call.
        exps = torch.exp(logs-logs.max(0).values)
        result.append((vals*exps).sum(0)/exps.sum(0))
    return torch.stack(result) if result else kv.new_empty((0, d))


def index_scores(q, k, weights):
    return ((q @ k.T).relu()*weights.unsqueeze(-1)).sum(-2)


def stable_topk(scores, count):
    return torch.argsort(scores, descending=True, stable=True)[..., :min(count, scores.shape[-1])]


def gathered_attention(q, kv, sink):
    scores = q @ kv.float().T / math.sqrt(q.shape[-1])
    logits = torch.cat((scores, sink[:, None]), dim=-1)
    probabilities = logits.softmax(-1)
    return probabilities[:, :-1] @ kv.float(), probabilities[:, -1]


def masked_attention_reference(q, all_kv, selected, sink):
    """Independent full matrix with -inf mask; used only for correctness, not performance."""
    scores = q @ all_kv.float().T / math.sqrt(q.shape[-1])
    mask = torch.ones(len(all_kv), dtype=torch.bool)
    mask[selected] = False
    scores[:, mask] = -torch.inf
    maximum = torch.maximum(scores.amax(-1), sink)
    exps = torch.exp(scores-maximum[:, None])
    denominator = exps.sum(-1)+torch.exp(sink-maximum)
    return exps @ all_kv.float()/denominator[:, None]


class AttentionComponent:
    def __init__(self, checkpoint, layer=2, precision='dequant_fp32'):
        if precision not in {'dequant_fp32', 'dynamic_int8_reference'}:
            raise ValueError('Unsupported precision')
        self.checkpoint, self.layer, self.precision = checkpoint, layer, precision
        self.cfg = checkpoint.config
        self.ratio = self.cfg['compress_ratios'][layer]
        if self.ratio not in {0, 4, 128}:
            raise ValueError('Unsupported ratio')
        self.prefix = f'layers.{layer}.attn.'
        self.matrices, self.quantized, self.params = {}, {}, {}
        names = ['wq_a', 'wq_b', 'wkv', 'wo_a', 'wo_b']
        params = ['q_norm.weight', 'kv_norm.weight', 'attn_sink']
        if self.ratio:
            names += ['compressor.wkv', 'compressor.wgate']
            params += ['compressor.ape', 'compressor.norm.weight']
        if self.ratio == 4:
            names += ['indexer.wq_b', 'indexer.weights_proj', 'indexer.compressor.wkv', 'indexer.compressor.wgate']
            params += ['indexer.compressor.ape', 'indexer.compressor.norm.weight']
        for name in names:
            self.matrices[name], self.quantized[name] = checkpoint.matrix(self.prefix+name)
        for name in params:
            self.params[name] = checkpoint.tensor(self.prefix+name).float()
        self.inv_freq = inverse_frequencies(self.cfg, bool(self.ratio))

    def linear(self, x, name):
        if self.quantized[name] and self.precision == 'dynamic_int8_reference':
            x = int8_reference(x)
        return F.linear(x, self.matrices[name])

    def compress(self, x, name='compressor', *, check_incremental=True):
        kv, gate = self.linear(x, name+'.wkv'), self.linear(x, name+'.wgate')
        ape = self.params[name+'.ape']
        pooled = pool_projected(kv, gate, ape, self.ratio)
        error = 0.0
        if check_incremental:
            incremental = pool_incremental(kv, gate, ape, self.ratio)
            error = float((pooled-incremental).abs().max()) if len(pooled) else 0.0
            torch.testing.assert_close(pooled, incremental, atol=2e-5, rtol=2e-5)
        pooled = rms(pooled, self.params[name+'.norm.weight'], self.cfg['rms_norm_eps'])
        pooled = rope(pooled, torch.arange(len(pooled))*self.ratio, self.inv_freq)
        if name.startswith('indexer'):
            pooled = hadamard(pooled)
            if self.precision == 'dynamic_int8_reference':
                pooled = int8_reference(pooled)
            return pooled, error
        return pooled.to(torch.bfloat16), error

    @torch.inference_mode()
    def prepare(self, hidden, prompt_tokens):
        """Cache-building prefill + teacher-specified decode hidden states; no text generation."""
        if hidden.ndim != 2 or hidden.shape[1] != self.cfg['hidden_size'] or not 0 < prompt_tokens < len(hidden):
            raise ValueError('Invalid hidden-state input')
        positions = torch.arange(len(hidden))
        raw_kv = rms(self.linear(hidden, 'wkv'), self.params['kv_norm.weight'], self.cfg['rms_norm_eps'])
        swa = rope(raw_kv, positions, self.inv_freq).to(torch.bfloat16)
        query_hidden = hidden[prompt_tokens:]
        qr = rms(self.linear(query_hidden, 'wq_a'), self.params['q_norm.weight'], self.cfg['rms_norm_eps'])
        q = self.linear(qr, 'wq_b').reshape(-1, self.cfg['num_attention_heads'], self.cfg['head_dim'])
        q = rms(q, eps=self.cfg['rms_norm_eps'])
        q = rope(q, positions[prompt_tokens:], self.inv_freq)
        compressed = hidden.new_empty((0, self.cfg['head_dim']), dtype=torch.bfloat16)
        check = {}
        if self.ratio:
            compressed, check['compressor_batch_vs_stream_absmax'] = self.compress(hidden)
        ik = iq = weights = None
        if self.ratio == 4:
            ik, check['index_compressor_batch_vs_stream_absmax'] = self.compress(hidden, 'indexer.compressor')
            iq = self.linear(qr, 'indexer.wq_b').reshape(-1, self.cfg['index_n_heads'], self.cfg['index_head_dim'])
            iq = hadamard(rope(iq, positions[prompt_tokens:], self.inv_freq))
            if self.precision == 'dynamic_int8_reference':
                iq = int8_reference(iq)
            weights = self.linear(query_hidden, 'indexer.weights_proj') / math.sqrt(self.cfg['index_head_dim']*self.cfg['index_n_heads'])
        return dict(hidden=hidden, prompt_tokens=prompt_tokens, q=q, swa=swa, compressed=compressed,
                    index_q=iq, index_k=ik, index_weights=weights, checks=check)

    def select(self, state, query):
        position = state['prompt_tokens']+query
        available = (position+1)//self.ratio if self.ratio else 0
        if self.ratio == 4:
            scores = index_scores(state['index_q'][query], state['index_k'][:available], state['index_weights'][query])
            selected = stable_topk(scores, self.cfg['index_topk'])
        else:
            scores = None
            selected = torch.arange(available)
        return selected, scores

    def output_projection(self, output, position):
        output = rope(output[None], torch.tensor([position]), self.inv_freq, inverse=True)[0]
        groups = self.cfg['o_groups']
        vector = output.reshape(groups, -1)
        matrix = self.matrices['wo_a'].reshape(groups, self.cfg['o_lora_rank'], -1)
        projected = torch.einsum('gd,grd->gr', vector, matrix).reshape(-1)
        return self.linear(projected, 'wo_b')

    def attend(self, state, query, selected, *, supplied=None, compare_dense=False):
        position = state['prompt_tokens']+query
        start = max(0, position-self.cfg['sliding_window']+1)
        swa = state['swa'][start:position+1]
        compressed = state['compressed'][selected] if supplied is None else supplied
        kv = torch.cat((swa, compressed), dim=0)
        output, sink_mass = gathered_attention(state['q'][query], kv, self.params['attn_sink'])
        error = None
        if compare_dense:
            available = (position+1)//self.ratio if self.ratio else 0
            full = torch.cat((state['swa'][:position+1], state['compressed'][:available]))
            allowed = torch.cat((torch.arange(start, position+1), selected+position+1))
            dense = masked_attention_reference(state['q'][query], full, allowed, self.params['attn_sink'])
            error = float((output-dense).abs().max())
            torch.testing.assert_close(output, dense, atol=2e-5, rtol=3e-5)
        return self.output_projection(output, position), dict(dense_mask_absmax=error,
            sink_probability_mean=float(sink_mass.mean()), read_rows=len(kv),
            swa_rows=len(swa), compressed_rows=len(compressed))


class TensorLRU:
    """Actual CPU tensor payloads stand in for bounded device slots; host is separate.

    Logical capacity is enforced. This is not real device memory or DMA.
    """
    def __init__(self, capacity, host):
        self.capacity, self.host = capacity, host
        self.resident = OrderedDict()
        self.prefetch = self.demand = self.generated = self.peak = 0

    def insert(self, block, protected=frozenset(), kind='prefetch'):
        if block in self.resident:
            return
        if len(self.resident) == self.capacity:
            victim = next((v for v in self.resident if v not in protected), None)
            if victim is None:
                raise ValueError('Working set exceeds capacity')
            del self.resident[victim]
        self.resident[block] = self.host[block].clone()
        setattr(self, kind, getattr(self, kind)+1)
        self.peak = max(self.peak, len(self.resident))

    def recall(self, selected):
        protected = set(selected)
        if len(protected) > self.capacity:
            raise ValueError('Working set exceeds capacity')
        for block in sorted(protected):
            self.insert(block, protected, 'demand')
            self.resident.move_to_end(block)
        return torch.stack([self.resident[b] for b in selected]) if selected else self.host[:0]


def controlled_hidden(seed, prompt_tokens, queries, dim=4096, shift=False):
    """Manufactured layer inputs; these are NOT activations produced by preceding layers."""
    g = torch.Generator().manual_seed(seed)
    prototypes = torch.randn(8, dim, generator=g)
    region_ids = torch.randint(0, 8, (math.ceil(prompt_tokens/128),), generator=g).repeat_interleave(128)[:prompt_tokens]
    decode_ids = torch.zeros(queries, dtype=torch.long)
    if shift:
        decode_ids[queries//2:] = 5
    ids = torch.cat((region_ids, decode_ids))
    noise = torch.randn(prompt_tokens+queries, dim, generator=g)
    x = rms(.7*prototypes[ids]+.3*noise)
    return x
