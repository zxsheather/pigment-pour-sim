"""Check that small updates accumulate against an FP32 AdamW reference."""
import unittest

import torch

from master_optimizer import FP32MasterAdamW


class MasterOptimizerTest(unittest.TestCase):
    def test_small_updates_match_full_precision_reference(self):
        model=torch.nn.Parameter(torch.tensor([.02,-.03,.5],dtype=torch.bfloat16))
        reference=torch.nn.Parameter(model.detach().float().clone())
        original=torch.optim.AdamW([model],lr=1e-5)
        optimizer=FP32MasterAdamW(original)
        expected=torch.optim.AdamW([reference],lr=1e-5)
        initial=model.detach().clone()
        for step in range(100):
            gradient=torch.tensor([1.,-1.,.5],dtype=torch.bfloat16)
            model.grad=gradient.clone()
            reference.grad=gradient.float()
            optimizer.step();expected.step()
            optimizer.zero_grad();expected.zero_grad()
        torch.testing.assert_close(model,reference.detach().to(torch.bfloat16),rtol=0,atol=0)
        self.assertFalse(torch.equal(model,initial))
        self.assertIsNone(model.grad)
        self.assertTrue(all(p.dtype==torch.float32 for g in optimizer.param_groups for p in g["params"]))

    def test_float32_parameters_and_zeroing_remain_supported(self):
        parameter=torch.nn.Parameter(torch.tensor([.02],dtype=torch.float32))
        optimizer=FP32MasterAdamW(torch.optim.AdamW([parameter],lr=1e-5))
        parameter.grad=torch.ones_like(parameter)
        optimizer.step();optimizer.zero_grad(set_to_none=False)
        self.assertLess(parameter.item(),.02)
        torch.testing.assert_close(parameter.grad,torch.zeros_like(parameter))


if __name__=="__main__":
    unittest.main()
