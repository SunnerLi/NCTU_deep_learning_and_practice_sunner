from torch.autograd import *
import misc.utils as utils
import torch.nn.functional as F
import torch.nn as nn
import torch

class CaptionModel(nn.Module):
    def __init__(self):
        super(CaptionModel, self).__init__()

    def beam_step(self, logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprob_sum, state):
        """
            Single step of beam search
        """
        ys, ix = torch.sort(logprobsf, 1, True)
        candidate = []
        cols = min(beam_size, ys.size(1))
        rows = beam_size

        # Calculate log probability for all possible sequence combination
        if t == 0:
            rows = 1
        for c in range(cols):
            for q in range(rows):
                local_logprob = ys[q, c]
                candidate_logprob = beam_logprob_sum[q] + local_logprob
                candidate.append({'c': ix[q, c], 'q': q, 'p': candidate_logprob, 'r': local_logprob})
        candidate = sorted(candidate, key = lambda x: -x['p'])

        # Obtain the sequence with highest log probability and update
        new_state = [_.clone() for _ in state]
        if t >= 1:
            beam_seq_prev = beam_seq[:t].clone()
            beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
        for vix in range(beam_size):
            v = candidate[vix]
            if t >= 1:
                beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
            for state_ix in range(len(new_state)):
                new_state[state_ix][:, vix] = state[state_ix][:, v['q']]
            beam_seq[t, vix] = v['c']
            beam_seq_logprobs[t, vix] = v['r']
            beam_logprob_sum[t, vix] = v['p']
        state = new_state
        return beam_seq, beam_seq_logprobs, beam_logprob_sum, state, candidate

    def beam_search(self, state, logprobs, *args, opt):
        # !!!
        
        beam_size = opt.beam_size
        beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
        beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_seq).zero_()
        beam_seq_logprobs_sum = torch.zeros(beam_size)
        done_beams = []
        
        # Do beam search for each time step
        for t in range(self.seq_length):
            logprobsf = logprobs.float()
            logprobsf[:, logprobsf.size(1) - 1] = logprobsf[:, logprobsf.size(1) - 1] - 1000
            beam_seq, beam_seq_logprobs, beam_seq_logprobs_sum, state, candidate = self.beam_step(
                logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_seq_logprobs_sum, state
            )
            for vix in range(beam_size):
                if beam_seq[t, vix] == 0 or t == self.seq_length - 1:
                    final_beam = {
                        'seq': beam_seq[:, vix].clone(),
                        'logps': beam_seq_logprobs[:, vix].clone(),
                        'p': beam_seq_logprobs_sum[vix]
                    }
                    done_beams.append(final_beam)
                    beam_seq_logprobs_sum[vix] = -1000
            it = beam_seq[t]
            logprob, state = self.get_logprob_state(Variable(it.cuda()), *(args + (state, )))
        
        done_beams = sorted(done_beams, key = lambda x: -x['p]'])[:beam_size]
        return done_beams
