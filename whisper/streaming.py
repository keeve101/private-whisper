from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from .tokenizer import Tokenizer
from .model import MonotonicTextDecoder
from simuleval.agents import GenericAgent, AgentStates
from simuleval.agents.actions import Action, ReadAction, WriteAction
from simuleval.data.segments import Segment, TextSegment
from torch import Tensor

class DecoderAgentStates(AgentStates):  # type: ignore
    def reset(self) -> None:
        self.source_len = 0
        self.target_indices: List[int] = []
        self.tgt_lang = None
        self.ngram_block_count = 0
        super().reset()

    def update_source(self, segment: Segment) -> None:
        """
        Update states from input segment
        Additionlly update incremental states

        Args:
            segment (~simuleval.agents.segments.Segment): input segment
        """
        self.source_finished = segment.finished
        if self.tgt_lang is None and segment.tgt_lang is not None:
            self.tgt_lang = segment.tgt_lang
        if not segment.is_empty:
            self.source = segment.content
            if len(self.source) == 0 and segment.finished:
                self.target_finished = True
                return
            self.source_len = self.source.size(1)

    def update_target(self, segment: Segment) -> None:
        """An AgentStates impl which doesn't update states.target"""
        self.target_finished = segment.finished
