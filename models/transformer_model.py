# -*- coding: utf-8 -*-
import random
import torch

from models.word_model import CaptionModel

class TransformerModel(CaptionModel):

    def __init__(self, encoder, decoder, **kwargs):
        super(TransformerModel, self).__init__(encoder, decoder, **kwargs)

    def train_forward(self, encoded, caps, cap_lens, **kwargs):
        # print(kwargs["ss_ratio"])
        if kwargs["ss_ratio"] != 1: # scheduled sampling training
            return self.stepwise_forward(encoded, caps, cap_lens, **kwargs)
        cap_max_len = caps.size(1)
        output = {}
        self.prepare_output(encoded, output, cap_max_len - 1)
        caps_padding_mask = (caps == self.pad_idx).to(encoded["audio_embeds"].device)
        caps_padding_mask = caps_padding_mask[:, :-1]
        decoder_output = self.decoder(words=caps[:, :-1], 
                                      enc_mem=encoded["audio_embeds"],
                                      enc_mem_lens=encoded["audio_embeds_lens"],
                                      caps_padding_mask=caps_padding_mask)
        self.train_process(output, decoder_output, cap_lens)
        return output

    def decode_step(self, decoder_input, encoded, caps, output, t, **kwargs):
        """Decoding operation of timestep t"""
        self.prepare_decoder_input(decoder_input, encoded, caps, output, t, **kwargs)
        # feed to the decoder to get states and logits
        output_t = self.decoder(**decoder_input)
        logits_t = output_t["logits"][:, -1, :]
        output_t["logits"] = logits_t.unsqueeze(1)
        outputs_t = output_t["outputs"][:,-1,:]
        output_t["outputs"] = outputs_t.unsqueeze(1)
        # sample the next input word and get the corresponding logits
        sampled = self.sample_next_word(logits_t, **kwargs)
        self.stepwise_process_step(output, output_t, t, sampled)

    ###
    ### add stepwise_process_step method
    def stepwise_process_step(self, output, output_t, t, sampled):
        """Postprocessing (save output values) after each timestep t"""
        super(TransformerModel, self).stepwise_process_step(output, output_t, t, sampled)
        output["logits"][:, t, :] = output_t["logits"].squeeze(1)
        output["seqs"][:, t] = sampled["w_t"]
        output["sampled_logprobs"][:, t] = sampled["probs"]
        # output["state"] = output_t["states"]
        output["outputs"][:, t, :] = output_t["outputs"].squeeze(1)
        # print(output_t["output"].shape)
    
    ###
    ### add prepare_output method
    def prepare_output(self, encoded, output, max_length):
        super(TransformerModel, self).prepare_output(encoded, output, max_length)
        output["outputs"] = torch.empty(output["seqs"].size(0), max_length,512)


    def prepare_decoder_input(self, decoder_input, encoded, caps, output, t, **kwargs):
        """Prepare the input dict `decoder_input` for the decoder and timestep t"""
        super(TransformerModel, self).prepare_decoder_input(decoder_input, encoded, caps, output, t, **kwargs)
        if t == 0:
            decoder_input["enc_mem"] = encoded["audio_embeds"]
            decoder_input["enc_mem_lens"] = encoded["audio_embeds_lens"]
            words = torch.tensor([self.start_idx,] * output["seqs"].size(0)).unsqueeze(1).long()
        else:
            words = output["seqs"][:, :t]
            if caps is not None and random.random() < kwargs["ss_ratio"]: # training, scheduled sampling
                words = caps[:, :t]
        decoder_input["words"] = words
        caps_padding_mask = (words == self.pad_idx).to(encoded["audio_embeds"].device)
        decoder_input["caps_padding_mask"] = caps_padding_mask

    def beamsearch_step(self, decoder_input, encoded, output, i, t, beam_size):
        self.prepare_beamsearch_decoder_input(decoder_input, encoded, output, i, t, beam_size)
        output_t = self.decoder(**decoder_input)
        output_t["logits"] = output_t["logits"][:, -1, :].unsqueeze(1)
        return output_t

    def prepare_beamsearch_decoder_input(self, decoder_input, encoded, output, i, t, beam_size):
        if t == 0:
            enc_mem_lens = encoded["audio_embeds_lens"][i]
            decoder_input["enc_mem_lens"] = enc_mem_lens.repeat(beam_size)
            enc_mem = encoded["audio_embeds"][i, :enc_mem_lens]
            decoder_input["enc_mem"] = enc_mem.unsqueeze(0).repeat(beam_size, 1, 1)

            words = torch.tensor([self.start_idx,] * beam_size).unsqueeze(1).long()
        else:
            words = output["seqs"]
        decoder_input["words"] = words
        caps_padding_mask = (words == self.pad_idx).to(encoded["audio_embeds"].device)
        decoder_input["caps_padding_mask"] = caps_padding_mask

    # def prepare_output(self, encoded, output, max_length):
    #     N = encoded["audio_embeds"].size(0)
    #     output["seqs"] = torch.empty(N, max_length, dtype=torch.long).fill_(self.end_idx)
    #     output["logits"] = torch.empty(N, max_length, self.vocab_size).to(encoded["audio_embeds"].device)
    #     output["outputs"] = torch.empty(output["seqs"].size(0), max_length, self.decoder.hidden_size).to(encoded["audio_embeds"].device)
    #     output["sampled_logprobs"] = torch.zeros(output["seqs"].size(0), max_length)
        
    # def stepwise_process_step(self, output, output_t, t, sampled):
    #     """Postprocessing (save output values) after each timestep t"""
    #     output["logits"][:, t, :] = output_t["logits"].squeeze(1)
    #     # output["outputs"][:, t, :] = output_t["output"]
    #     output["seqs"][:, t] = sampled["w_t"]
    #     output["sampled_logprobs"][:, t] = sampled["probs"]
    #     # output["state"] = output_t["states"]