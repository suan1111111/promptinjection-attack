import unittest
import torch
from validate_frequency_sparse import propose_sparse, swap_fraction


class SearchTests(unittest.TestCase):
    def test_proposals_preserve_exact_budget_and_do_not_modify_input(self):
        clean=torch.full((1,3,28,28),.5)
        original=clean.clone()
        gen=torch.Generator().manual_seed(17)
        support=torch.arange(20)
        best=clean.clone()
        best.reshape(1,3,-1)[:,:,support]=0
        for count in (1,5,20):
            candidate,updated=propose_sparse(clean,best,support,count,gen,torch)
            self.assertEqual(len(updated.unique()),20)
            self.assertEqual(int(((candidate-clean).abs().amax(1)>0).sum()),20)
            self.assertEqual(len(set(updated.tolist())-set(support.tolist())),count)
            self.assertTrue(torch.all((candidate==0)|(candidate==.5)|(candidate==1)))
            torch.testing.assert_close(clean,original)
            best,support=candidate,updated

    def test_schedule_decreases_and_stays_positive(self):
        values=[swap_fraction(i,200) for i in range(1,201)]
        self.assertEqual(values,sorted(values,reverse=True))
        self.assertGreater(min(values),0)


if __name__=='__main__':
    unittest.main()
