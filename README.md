# GPU poor ESMFold through Hugging Face and Quanto
ESMFold is a protein language model based on the ESM-2 3B parameter model developed by the Meta Fundamental AI Research Protein Team (FAIR) ([paper](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v2)).
It is one of the best model available when it comes to predicting the structure of a protein from the amino acids sequence. However, the GPU resources that are necessary to run this model can be prohibitive, even for sequences of a few hundreds of residues. This article aim at finding some possible solutions to overcome this issue, mainly using quantization techniques.

## Usage
To get started with this model you can either follow the instructions on the ESM GitHub page ([github](https://github.com/facebookresearch/esm?tab=readme-ov-file#esmfold)) or use the [Huggin Face Tranformer library](https://huggingface.co/docs/transformers/model_doc/esm), which provides an easy-to-use implementation and doesn't require the ESMFold dependencies. 

In order to use the tranformer library you need to install the `transformers` and `accelerate` libraries
```
pip install transformers accelerate
```
Predicting the protein structure for a given sequence is then straightforward. First you need to instantiate the model, eventually transferring the model to GPU

```python
from transformers import AutoTokenizer, EsmForProteinFolding

tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)

model = model.cuda()
```
Then given a `sequence` it is possible to create the model inputs through the `tokenizer`

```python
sequence="SHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK"

inputs = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)['input_ids']
```
Finally, getting the model outputs is as simple as 
```python
import torch

with torch.no_grad():
    output = model(inputs)
```
The `output` contains differrent protein informations, e.g the atom positions 

```python
positions = outputs['positions'][-1, 0]

```


## Optimization
The main problem with the above implementation is the GPU memory demand: loading the model through the Hugging Face `transformers` requires at least 15GB of GPU memory, which can grow as far as 24GB when considering longer sequences. 
For instance, for the previously considered protein, which is made of 256 residues ([pdb](https://www.rcsb.org/structure/1CA2)), `nvidia_smi` indicates that 16GB of memory where used for the inference.<br/>
Hence, the first possibility to make the model lighter is to convert the model weights to `float16`, a sort of post-training quantization. This is done with a single line of code

```python
model.esm = model.esm.half()
```
With this optimization of the model performance and requirements the memory usage is significantily decreased to 10.2GB, with little or no variation in the inference accuracy.


## Quantization
Quantization is a powerful tool when it come to achieving lower GPU memory requirements to run deep learning models. In particular, post training quantization (PTQ) aims at reducing the necessary resources for inference converting the wheights and the activations of the model to another type, e.g. `float8` or `int8`, while in principle preserving the model accuracy. To this end, it is possibile to use the python quantization toolkit `Quanto`([github](https://github.com/huggingface/quanto)), which is integrated in the Hugging Face `transformers` library and it is straightforward to implement. 

```python
from transformers import AutoTokenizer, EsmForProteinFolding, QuantoConfig

tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
quantization_config = QuantoConfig(weights="int8")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", device_map="cuda", quantization_config=quantization_config)

```



6GB


![alt text](https://github.com/davideaguglia/ESMFold/blob/ef0ad408b26dee7d15755805e21ac5e3a6329a03/plots/acc.png)

![alt text](https://github.com/davideaguglia/ESMFold/blob/ef0ad408b26dee7d15755805e21ac5e3a6329a03/plots/memory.png)

![alt text](https://github.com/davideaguglia/ESMFold/blob/ef0ad408b26dee7d15755805e21ac5e3a6329a03/plots/time.png)

![alt text](https://github.com/davideaguglia/ESMFold/blob/0f91aea9c07d44897b11eb094b268c319e0f2dde/plots/models.png)
