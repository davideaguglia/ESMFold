# GPU poor ESMFold through Hugging Face and Quanto
ESMFold is one of the best model available when it comes to predicting the structure of a protein from the amino acids sequence. It is a protein language model based on the ESM-2 3B parameter model developed by the Meta Fundamental AI Research Protein Team (FAIR) ([paper](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v2))

'''

from transformers import AutoTokenizer, EsmForProteinFolding

tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)

model = model.cuda()


'''
