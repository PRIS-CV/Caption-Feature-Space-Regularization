import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import get_centroids, get_cossim, calc_loss
from utils.train_util import mean_with_lens, max_with_lens

class Stage1Encoder(nn.Module):
    def __init__(self,vocab_size,**kwargs):
        super(Stage1Encoder,self).__init__()
        self.inputdim = kwargs.get("inputdim",512)
        self.embed_size = kwargs.get("embed_size",512)
        self.hidden_size = kwargs.get('hidden_size', 512)
        self.bidirectional = kwargs.get('bidirectional', False)
        self.num_layers = kwargs.get('num_layers', 1)
        self.dropout = kwargs.get('dropout', 0.3)
        self.vocab_size =  vocab_size

        self.word_embeddings = nn.Embedding(self.vocab_size, self.inputdim)
        self.lstm = nn.LSTM(self.inputdim,self.hidden_size,self.num_layers,batch_first=True,dropout=self.dropout,bidirectional=self.bidirectional)
        self.outputlayer = nn.Linear(self.hidden_size*(self.bidirectional + 1),self.embed_size)
        self.bn_outputlayer = nn.BatchNorm1d(self.embed_size, momentum=0.1, affine=False)
        
        self.embedding = nn.Linear(self.embed_size, self.embed_size)
        # self.dropoutlayer = nn.Dropout(self.dropout)
    
    def init(self):
        for m in self.modules():
            m.apply(self.init_weights)
            
    def init_weights(self, m):
        if isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m,nn.Embedding):
            nn.init.kaiming_uniform_(m.weight)

                
    def forward(self,*input):
        x,lens = input
          
        x = self.word_embeddings(x)
        lens = torch.as_tensor(lens)
        packed = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        packed_out, hid = self.lstm(packed)
        packed_out,lens = nn.utils.rnn.pad_packed_sequence(packed_out,batch_first=True)
        # print(packed_out.shape)
        packed_out = self.outputlayer(packed_out)
        packed_out = packed_out.transpose(1,2)
        packed_out = self.bn_outputlayer(packed_out)
        packed_out = packed_out.transpose(1,2)
    
        x_mean = mean_with_lens(packed_out, lens)
        x_max = max_with_lens(packed_out, lens)
        stats = x_mean + x_max
        
        out = F.dropout(stats, p=0.5, training=self.training)
        out = self.embedding(out)
        
        return {
            "caption_embeds": out,
            "audio_embeds_lens": lens}

    def load_word_embeddings(self, embeddings, tune=True, **kwargs):
        assert embeddings.shape[0] == self.vocab_size, "vocabulary size mismatch!"
    
        embeddings = torch.as_tensor(embeddings).float()
        self.word_embeddings.weight = nn.Parameter(embeddings)
        for para in self.word_embeddings.parameters():
            para.requires_grad = tune

        if embeddings.shape[1] != self.inputdim:
            assert "projection" in kwargs, "embedding size mismatch!"
        if kwargs["projection"]:
            self.word_embeddings = nn.Sequential(
                self.word_embeddings,
                nn.Linear(embeddings.shape[1], self.inputdim)
            )
        
    
class GE2ELoss(nn.Module):
    
    def __init__(self, device):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(10.0).to(device), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(-5.0).to(device), requires_grad=True)
        self.device = device
        
    def forward(self, embeddings):
        torch.clamp(self.w, 1e-6)
        centroids = get_centroids(embeddings)
        cossim = get_cossim(embeddings, centroids)
        sim_matrix = self.w*cossim.to(self.device) + self.b
        loss, _ = calc_loss(sim_matrix)
        return loss

