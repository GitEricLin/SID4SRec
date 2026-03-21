# B
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import TransformerEncoder, info_nce
from transformers.models.deberta.modeling_deberta import DebertaEncoder
from torch.nn import  CrossEntropyLoss
import torch as th
from utils import (
    SiLU,
    linear,
    timestep_embedding
)

class SID4SRec(nn.Module):

    def __init__(self, device, args):
        super(SID4SRec, self).__init__()
        self.item_size = args.item_size
        self.batch_size = args.train_batch_size
        self.hidden_size = args.hidden_size
        self.device = device
        self.args = args
        
        #************params for sasrec****************
        self.max_seq_length = args.max_seq_length
        self.n_layers = args.n_layers
        self.n_heads = args.n_heads
        self.inner_size = args.inner_size
        self.hidden_dropout_prob = args.sasrec_dropout_prob
        self.attn_dropout_prob = args.sasrec_dropout_prob
        self.hidden_act = args.hidden_act
        self.layer_norm_eps = args.layer_norm_eps
        self.initializer_range = args.initializer_range
        self.batch_size = args.train_batch_size
        self.item_contrastive = CategoryContrastiveLearning(self.device, args)
        
        
        self.item_embedding = torch.nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size * 3)

        self.nce_fct = nn.CrossEntropyLoss()
        self.rec_trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size * 3,
            inner_size=self.inner_size * 3,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )
        
        self.LayerNorm = nn.LayerNorm(self.hidden_size * 3, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
                
        self.hidden_t_dim = self.hidden_size
        time_embed_dim = self.hidden_size * 4
        self.time_embed = nn.Sequential(
            linear(self.hidden_size, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, self.hidden_size),
        )
        self.bert_encoder = DebertaEncoder(args,)
        self.ln_bert = nn.LayerNorm(args.hidden_size)
        self.bias = nn.Parameter(torch.zeros(self.item_size))
        self.bert_loss_fct = CrossEntropyLoss()  # -100 index = padding token
        self.logits_mode = 1
        self.register_buffer("position_ids", torch.arange(self.max_seq_length).expand((1, -1)))
        self.softmax_loss_fct = nn.CrossEntropyLoss()

        self.category_emb = torch.nn.Embedding(
            num_embeddings=self.args.n_categories,
            embedding_dim=self.args.hidden_size
        ).to(self.device)

        self.brand_emb = torch.nn.Embedding(
            num_embeddings=self.args.n_brands,
            embedding_dim=self.args.hidden_size
        ).to(self.device)

        # parameters initialization
        self.apply(self._init_weights)

    def get_att_emb(self, ids):
        categories = self.item_contrastive.get_item_categories(ids)
        cat_emb = self.category_emb(categories)
        brands = self.item_contrastive.get_item_brands(ids)  # New method needed in CategoryContrastiveLearning
        brand_emb = self.brand_emb(brands)
        return cat_emb, brand_emb

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def get_embeds(self, item_ids):
        return self.item_embedding(item_ids)

    def get_logits(self, hidden_repr):
        if self.logits_mode == 1:
            test_items_emb = self.item_embedding.weight
            scores = torch.matmul(hidden_repr, test_items_emb.transpose(0, 1))  # [B n_items+2]
            return scores
        elif self.logits_mode == 2: # standard cosine similarity
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)  # vocab
            text_emb_t = th.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * th.mm(self.lm_head.weight,
                                                                     text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
            scores = th.sqrt(th.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                               hidden_repr.size(1)) # vocab, bsz*seqlen
            scores = -scores.permute(1, 2, 0).contiguous()
            return scores
        else:
            raise NotImplementedError
        

    def diffusion_reverse(self, x, timesteps, attention_mask):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim))
        emb_x = x
    
        seq_length = x.size(1)
        position_ids = self.position_ids[:, : seq_length ]
      
        # emb_inputs = self.position_embedding(position_ids) + emb_x + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = emb_x + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.ln_bert(emb_inputs))

        input_trans_hidden_states = self.bert_encoder(emb_inputs,attention_mask,output_hidden_states=False,output_attentions=False,return_dict=False,)
        h = input_trans_hidden_states[0]
        # print(" debug: ", h.shape)
        h = h.type(x.dtype)
        return h

  
    
    def add_position_embedding(self, sequence, seq_emb=None):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        
        if seq_emb == None:
            item_embeddings = self.item_embedding(sequence)

        else:
            item_embeddings = seq_emb

        category_emb, brand_emb = self.get_att_emb(sequence)
        item_embeddings = torch.cat([item_embeddings, category_emb, brand_emb], dim=-1)
        position_embeddings = self.position_embedding(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb
    

    def get_extended_attention_mask(self, input_ids):
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()
        subsequent_mask = subsequent_mask.cuda().to(input_ids.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


    def forward(self, seq_emb, extended_attention_mask):
        item_encoded_layers = self.rec_trm_encoder(seq_emb, extended_attention_mask, output_all_encoded_layers=True)
        sequence_output = item_encoded_layers[-1]
        return sequence_output
    
    
    def calculate_rec_loss(self, item_seq, target_pos, target_neg):
        extended_attention_mask = self.get_extended_attention_mask(item_seq)
        sequence_emb = self.add_position_embedding(item_seq)
        seq_output = self.forward(sequence_emb, extended_attention_mask)
        loss = self.cross_entropy(seq_output, target_pos, target_neg)

        # test_item_emb = self.item_embedding.weight
        # logits = torch.matmul(seq_output[:,-1,:], test_item_emb.transpose(0, 1))
        # pos_items = target_pos[:, -1]
        # loss = self.softmax_loss_fct(logits, pos_items)

        return loss
    
    
    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_item_emb = self.item_embedding(pos_ids)
        neg_item_emb = self.item_embedding(neg_ids)
        
        pos_cat_emb, pos_brand_emb = self.get_att_emb(pos_ids)
        neg_cat_emb, neg_brand_emb = self.get_att_emb(neg_ids)
        
        pos_emb = torch.cat([pos_item_emb, pos_cat_emb, pos_brand_emb], dim=-1)
        neg_emb = torch.cat([neg_item_emb, neg_cat_emb, neg_brand_emb], dim=-1)

        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))

        seq_emb = seq_out.view(-1, self.hidden_size * 3)  # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.max_seq_length).float()  # [batch*seq_len]
        # loss = torch.sum(
        #     -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
        #     - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        # ) / torch.sum(istarget)
        all_items = torch.arange(self.args.item_size).to(self.device)
        items_emb = self.item_embedding.weight
        all_categories = self.item_contrastive.get_item_categories(all_items.unsqueeze(0))
        all_brands = self.item_contrastive.get_item_brands(all_items.unsqueeze(0))
        cat_emb = self.category_emb(all_categories[0])
        brand_emb = self.brand_emb(all_brands[0])
        test_items_emb = torch.cat([items_emb, cat_emb, brand_emb], dim=-1)
        logits = torch.matmul(seq_emb, test_items_emb.transpose(0, 1))
        loss = self.nce_fct(logits, torch.squeeze(pos_ids.view(-1)))

        return loss


    def full_sort_predict(self, item_seq):
        extended_attention_mask = self.get_extended_attention_mask(item_seq)
        sequence_emb = self.add_position_embedding(item_seq)
        seq_output = self.forward(sequence_emb, extended_attention_mask)
        seq_output = seq_output[:,-1,:]

        all_items = torch.arange(self.args.item_size).to(self.device)
        items_emb = self.item_embedding.weight
        
        all_categories = self.item_contrastive.get_item_categories(all_items.unsqueeze(0))
        all_brands = self.item_contrastive.get_item_brands(all_items.unsqueeze(0))
        cat_emb = self.category_emb(all_categories[0])
        brand_emb = self.brand_emb(all_brands[0])
        test_items_emb = torch.cat([items_emb, cat_emb, brand_emb], dim=-1)
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1)) 
        return scores


    def calculate_cl_loss(self, aug_seq1, aug_seq2, emb1=None, emb2=None):
        extended_attention_mask = self.get_extended_attention_mask(aug_seq1)
        aug_seq1_full_emb = self.add_position_embedding(aug_seq1, emb1)
        aug_seq2_full_emb = self.add_position_embedding(aug_seq2, emb2)

        seq1_output = self.forward(aug_seq1_full_emb, extended_attention_mask)[:,-1,:]
        seq2_output = self.forward(aug_seq2_full_emb, extended_attention_mask)[:,-1,:]
        
        nce_logits, nce_labels = info_nce(seq1_output, seq2_output, temp=self.args.temperature, batch_size=aug_seq1.shape[0], sim="dot")
        nce_loss = self.nce_fct(nce_logits, nce_labels)
        return nce_loss
    
    def get_item_embeddings(self):
        return self.item_embedding.weight


class CategoryContrastiveLearning(nn.Module):
    def __init__(self, device, args):
        super().__init__()
        self.args = args
        self.temperature = args.temperature
        self.phi = args.phi
        self.item_to_category = args.item_to_category
        self.item_to_brand = args.item_to_brand
        self.category_items = args.category_items
        self.brand_items = args.brand_items
        self.item_temp = args.item_temp
        self.device = device

        self._category_lookup = args.category_lookup.to(device)
        self._brand_lookup = args.brand_lookup.to(device)

    def get_item_categories(self, item_ids):
        # 使用預先生成的 lookup 表
        flattened_ids = item_ids.flatten()
        valid_mask = flattened_ids < len(self._category_lookup)
        
        categories = torch.zeros_like(flattened_ids, device=self.device)
        if valid_mask.any():
            categories[valid_mask] = self._category_lookup[flattened_ids[valid_mask]]
        
        return categories.reshape(item_ids.shape)
    
    def get_item_brands(self, item_ids):
        # Similar logic for brand lookup
        flattened_ids = item_ids.flatten()
        valid_mask = flattened_ids < len(self._brand_lookup)
        
        brands = torch.zeros_like(flattened_ids, device=self.device)
        if valid_mask.any():
            brands[valid_mask] = self._brand_lookup[flattened_ids[valid_mask]]
        
        return brands.reshape(item_ids.shape)

    def get_category_pooling(self, item_embeddings):
        # 預先計算每個類別的項目索引和數量
        if not hasattr(self, '_category_indices'):
            self._category_indices = {}
            self._category_counts = {}
            for cat_id, items in self.category_items.items():
                if items:  # 只處理非空類別
                    self._category_indices[cat_id] = torch.tensor(items, device=self.device)
                    self._category_counts[cat_id] = len(items)
        
        # 直接使用快取的索引進行 pooling
        category_embeddings = {}
        for cat_id, indices in self._category_indices.items():
            indices = indices.to(self.device)  # 確保索引在正確的設備上
            cat_items = torch.index_select(item_embeddings, 0, indices)
            category_embeddings[cat_id] = torch.mean(cat_items, dim=0)
        
        return category_embeddings
    
    def compute_similarity(self, x, y):
        """計算兩個向量的余弦相似度"""
        x_norm = F.normalize(x, dim=-1)
        y_norm = F.normalize(y, dim=-1)
        return torch.sum(x_norm * y_norm, dim=-1)
    
    
    def forward(self, item_embeddings, batch_items=None):
        """前向傳播，計算對比損失"""
        return self.get_contrastive_loss(item_embeddings, batch_items)
    

    # def get_contrastive_loss(self, item_embeddings, batch_items=None):
    #     """計算帶實例加權的對比損失（向量化版本）"""
    #     category_embeddings = self.get_category_pooling(item_embeddings)
        
    #     # 如果沒有有效的類別，返回0
    #     if not category_embeddings:
    #         return torch.tensor(0.0, device=self.device)
        
    #     # 將category_embeddings轉換為tensor形式
    #     categories = list(category_embeddings.keys())
    #     category_embeds = torch.stack([category_embeddings[c] for c in categories])
        
    #     # 創建item到category的映射矩陣
    #     n_items = len(item_embeddings)
    #     n_categories = len(categories)
    #     item_category_mask = torch.zeros(n_items, n_categories, dtype=torch.bool, device=self.device)  # 明確指定dtype為bool
        
    #     # 使用索引操作建立mask
    #     if batch_items is not None:
    #         valid_items = torch.tensor(batch_items, device=self.device)
    #         valid_mask = torch.zeros(n_items, dtype=torch.bool, device=self.device)
    #         valid_mask[valid_items] = True
    #     else:
    #         valid_mask = torch.ones(n_items, dtype=torch.bool, device=self.device)
        
    #     # 建立item到category的對應關係矩陣
    #     category_indices = torch.tensor([self.item_to_category.get(i, -1) for i in range(n_items)], 
    #                                 device=self.device)
    #     for cat_idx, category in enumerate(categories):
    #         item_category_mask[category_indices == category] = True  # 使用True而不是1
        
    #     # 應用valid_mask
    #     item_category_mask = item_category_mask[valid_mask]
        
    #     # 只選擇有效的item embeddings
    #     valid_item_embeddings = item_embeddings[valid_mask]
        
        
    #     if len(valid_item_embeddings) == 0:
    #         return torch.tensor(0.0, device=self.device)
        
    #     # 計算所有item與所有category的相似度
    #     all_similarities = self.compute_similarity(
    #         valid_item_embeddings.unsqueeze(1),
    #         category_embeds.unsqueeze(0)
    #     )
                
    #     # 為每個item創建對應的負例mask
    #     neg_mask = ~item_category_mask  # 現在item_category_mask是bool類型
        
    #     # 計算實例權重 (確保轉換為float進行數值計算)
    #     # instance_weights = ((all_similarities < self.phi).float() * neg_mask.float())
    #     instance_weights = torch.where(
    #         all_similarities >= self.phi,
    #         torch.zeros_like(all_similarities),
    #         torch.ones_like(all_similarities)
    #     ) * neg_mask.float()

    #     all_similarities = all_similarities / self.item_temp
        
    #     # 獲取正例相似度和加權負例相似度
    #     pos_similarities = torch.sum(all_similarities * item_category_mask.float(), dim=1, keepdim=True)
    #     neg_similarities = all_similarities * instance_weights
        
    #     # 組合logits
    #     logits = torch.cat([pos_similarities, neg_similarities], dim=1)
        
    #     # 創建標籤（正例永遠是第一個）
    #     labels = torch.zeros(len(valid_item_embeddings), dtype=torch.long, device=self.device)
        
    #     # 計算loss
    #     loss = F.cross_entropy(logits, labels, reduction='none')
        
    #     # 根據item所屬的category數量進行加權
    #     weights = item_category_mask.sum(dim=1).float()
    #     weighted_loss = (loss * weights).sum() / weights.sum()
        
    #     return weighted_loss

    def get_contrastive_loss(self, item_embeddings, batch_items=None):
        """使用InfoNCE計算對比學習損失，使用phi閾值過濾負例（完全向量化版本）"""
        category_embeddings = self.get_category_pooling(item_embeddings)
        
        if not category_embeddings:
            return torch.tensor(0.0, device=self.device)
        
        # 將category_embeddings轉換為tensor形式
        categories = list(category_embeddings.keys())
        category_embeds = torch.stack([category_embeddings[c] for c in categories])
        
        # 建立類別ID到index的映射
        category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
        category_to_idx_tensor = torch.tensor(
            [category_to_idx.get(i, -1) for i in range(max(category_to_idx.keys()) + 1)],
            device=self.device
        )
        
        # 處理valid items
        if batch_items is not None:
            valid_items = torch.tensor(batch_items, device=self.device)
            valid_mask = torch.zeros(len(item_embeddings), dtype=torch.bool, device=self.device)
            valid_mask[valid_items] = True
        else:
            valid_mask = torch.ones(len(item_embeddings), dtype=torch.bool, device=self.device)
        
        # 獲取有效的embeddings和對應的類別
        valid_item_embeddings = item_embeddings[valid_mask]
        indices = torch.arange(len(item_embeddings), device=self.device)
        category_indices = torch.tensor([self.item_to_category.get(i.item(), -1) for i in indices[valid_mask]],
            device=self.device
        )
        
        if len(valid_item_embeddings) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # 正則化embeddings
        item_embeds_normalized = F.normalize(valid_item_embeddings, dim=1)
        category_embeds_normalized = F.normalize(category_embeds, dim=1)
        
        # 計算相似度矩陣 [batch_size, n_categories]
        raw_similarities = torch.matmul(item_embeds_normalized, category_embeds_normalized.t())
        
        # 創建正例mask - 使用高級索引
        batch_size = len(valid_item_embeddings)
        n_categories = len(categories)
        positive_mask = torch.zeros((batch_size, n_categories), dtype=torch.bool, device=self.device)
        valid_category_positions = category_to_idx_tensor[category_indices]
        valid_positions = valid_category_positions != -1
        batch_indices = torch.arange(batch_size, device=self.device)
        positive_mask[batch_indices[valid_positions], valid_category_positions[valid_positions]] = True
        
        # 創建負例mask（基於phi閾值）
        negative_mask = (~positive_mask) & (raw_similarities < self.phi)
        
        # 應用溫度係數的相似度
        similarities = raw_similarities / self.item_temp
        
        # 將不符合條件的負例設置為很大的負值
        similarities = torch.where(
            positive_mask | negative_mask,
            similarities,
            torch.tensor(-1e9, device=self.device)
        )
        
        # 創建標籤 - 使用argmax找到每個item的正例位置
        labels = valid_category_positions
        labels = torch.where(labels >= 0, labels, 0)  # 將無效的標籤設為0
        
        # 計算InfoNCE損失
        loss = F.cross_entropy(similarities, labels)
        
        # 添加監控指標
        # with torch.no_grad():
        #     n_total_pairs = (~positive_mask).sum().item()
        #     n_valid_negatives = negative_mask.sum().item()
        #     pos_sims = raw_similarities[positive_mask].mean().item()
        #     neg_sims = raw_similarities[negative_mask].mean().item() if negative_mask.sum() > 0 else 0
            
        #     print(f"\nContrastive Learning Statistics:")
        #     print(f"Total number of potential negative pairs: {n_total_pairs}")
        #     print(f"Number of valid negative pairs (after phi): {n_valid_negatives}")
        #     print(f"Ratio of valid negatives: {n_valid_negatives/n_total_pairs*100:.2f}%")
        #     print(f"Average positive similarity: {pos_sims:.4f}")
        #     print(f"Average negative similarity: {neg_sims:.4f}")
        #     print(f"Temperature: {self.item_temp}")
        #     print(f"Phi threshold: {self.phi}")
        
        return loss
    
    # def analyze_phi_effect(self, item_embeddings, item_id):
    #     """
    #     分析 phi 閾值對指定商品負例選擇的影響
        
    #     Args:
    #         item_embeddings: 所有商品的 embedding
    #         item_id: 要分析的商品 ID
    #     """
    #     print(f"\n===== 分析 phi={self.phi} 對 item{item_id} 負例選擇的影響 =====")
        
    #     # 獲取商品的 embedding
    #     item_emb = item_embeddings[item_id].unsqueeze(0)  # [1, hidden_size]
        
    #     # 計算相似度
    #     similarities = F.normalize(item_emb, dim=-1) @ F.normalize(item_embeddings, dim=-1).t()  # [1, n_items]
    #     similarities = similarities.squeeze()
        
    #     # 獲取商品的類別
    #     category_id = self.item_to_category.get(item_id)
        
    #     # 找出所有正例商品（同類別）
    #     positive_items = set()
    #     if category_id is not None:
    #         positive_items.update(self.category_items.get(category_id, []))
    #     positive_items.remove(item_id)
        
    #     # 所有其他商品都是潛在的負例
    #     all_items = set(range(len(item_embeddings)))
    #     negative_items = all_items - positive_items - {0, item_id}  # 排除padding item(0)和目標商品
        
    #     # 分析相似度分佈
    #     neg_sims = similarities[list(negative_items)]
    #     n_total_negs = len(negative_items)
    #     n_valid_negs = (neg_sims < self.phi).sum().item()
        
    #     print(f"\n1. 負例統計:")
    #     print(f"   - 總潛在負例數量: {n_total_negs}")
    #     print(f"   - 符合 phi 閾值的負例數量: {n_valid_negs}")
    #     print(f"   - 被 phi 過濾掉的負例數量: {n_total_negs - n_valid_negs}")
    #     print(f"   - 有效負例比例: {n_valid_negs/n_total_negs*100:.2f}%")
        
    #     print(f"\n2. 相似度分佈:")
    #     print(f"   - 最大相似度: {neg_sims.max().item():.4f}")
    #     print(f"   - 最小相似度: {neg_sims.min().item():.4f}")
    #     print(f"   - 平均相似度: {neg_sims.mean().item():.4f}")
        
    #     # 顯示一些被過濾掉的負例
    #     filtered_negs = torch.tensor(list(negative_items), device=self.device)[neg_sims >= self.phi]
    #     if len(filtered_negs) > 0:
    #         print(f"\n3. 被 phi 過濾掉的負例示例 (相似度 >= {self.phi}):")
    #         for i, neg_id in enumerate(filtered_negs[:5]):
    #             sim = similarities[neg_id].item()
    #             print(f"   - Item {neg_id}: 相似度 = {sim:.4f}")
