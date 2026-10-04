"""Versioned memory events used by the public task workload adapter.

Every mutation records the state hash before and after it.  The audit is
append-only; a replay can therefore distinguish a write from a consolidation
or a supersession without inferring the operation from prose.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


@dataclass
class Memory:
    task_id: str
    entries: dict[str, dict] = field(default_factory=dict)
    audit: list[dict] = field(default_factory=list)
    sequence: int = 0

    def _record(self, kind: str, key: str | None, before: dict, after: dict,
                *, actor: str, detail: dict | None = None) -> dict:
        self.sequence += 1
        event = {'sequence': self.sequence, 'event_type': kind,
                 'source': 'memory_module', 'actor': actor, 'task_id': self.task_id,
                 'key': key, 'before_sha256': digest(before), 'after_sha256': digest(after),
                 'before': before, 'after': after, 'detail': detail or {},
                 'time_ns': time.time_ns()}
        self.audit.append(event)
        return event

    def write(self, key: str, value: str, *, actor: str = 'agent') -> dict:
        if key in self.entries:
            raise ValueError('Existing key requires supersede')
        before = dict(self.entries)
        self.entries[key] = {'version': 1, 'value': value, 'sources': [self.task_id]}
        return self._record('memory_write', key, before, dict(self.entries), actor=actor)

    def consolidate(self, target: str, sources: list[str], value: str,
                    *, actor: str = 'agent') -> dict:
        if target in self.entries or len(set(sources)) < 2 or any(s not in self.entries for s in sources):
            raise ValueError('Consolidation requires two existing distinct keys and a new target')
        before = dict(self.entries)
        self.entries[target] = {'version': 1, 'value': value,
                                'sources': [f'{s}@{self.entries[s]["version"]}' for s in sources]}
        return self._record('memory_consolidation', target, before, dict(self.entries),
                            actor=actor, detail={'source_keys': sources})

    def supersede(self, key: str, value: str, *, actor: str = 'agent') -> dict:
        if key not in self.entries:
            raise ValueError('Supersession requires an existing key')
        before = dict(self.entries)
        old = self.entries[key]
        self.entries[key] = {'version': old['version'] + 1, 'value': value,
                             'sources': old['sources']}
        return self._record('memory_supersession', key, before, dict(self.entries), actor=actor,
                            detail={'replaced_version': old['version']})

    def switch_task(self, task_id: str, *, actor: str = 'harness') -> dict:
        if task_id == self.task_id:
            raise ValueError('Task switch must change task identity')
        before = {'task_id': self.task_id, 'entries': dict(self.entries)}
        self.task_id = task_id
        after = {'task_id': self.task_id, 'entries': dict(self.entries)}
        return self._record('task_switch', None, before, after, actor=actor)
