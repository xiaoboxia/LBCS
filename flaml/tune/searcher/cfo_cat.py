# !
#  * Copyright (c) Microsoft Corporation. All rights reserved.
#  * Licensed under the MIT License. See LICENSE file in the
#  * project root for license information.
from .flow2 import FLOW2
from .blendsearch import CFO


class FLOW2Cat(FLOW2):
    """Local search algorithm optimized for categorical variables."""
    STEP = None
    
    @classmethod
    def set_step(cls, step: float):
        cls.STEP = step
    
    def _init_search(self):
        super()._init_search()
        self.step_ub = 20
        if self.STEP is not None:
            self.step = self.STEP
        else:
            self.step = 8
        lb = self.step_lower_bound
        if lb > self.step:
            self.step = lb * 2
        # upper bound
        if self.step > self.step_ub:
            self.step = self.step_ub
        # self._trunc = self.dim


class CFOCat(CFO):
    """CFO optimized for categorical variables."""

    LocalSearch = FLOW2Cat