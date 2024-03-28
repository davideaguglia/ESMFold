# ESMFold for GPU poor through Hugging Face and Quanto
ESMFold is a protein language model based on the ESM-2 3B parameter architecture developed by the Meta Fundamental AI Research Protein Team (FAIR) ([paper](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v2)).
It is one of the best model available when it comes to predicting the structure of a protein from the amino acids sequence. However, the GPU resources that are necessary to run this model can be prohibitive, even for sequences of a few hundreds of residues. This article aim at finding some possible solutions to overcome this issue, using quantization techniques.

## Usage
To get started with this model you can either follow the instructions on the ESM GitHub page ([github](https://github.com/facebookresearch/esm?tab=readme-ov-file#esmfold)) or use the [Huggin Face Transformers library](https://huggingface.co/docs/transformers/model_doc/esm), which provides an easy-to-use implementation and doesn't require the ESMFold dependencies. 

In order to use the tranformer library you need to install `transformers` and `accelerate` 
```
pip install transformers accelerate
```
Predicting the protein structure for a given sequence is then straightforward. First you need to instantiate the model, eventually transferring it to GPU

```python
from transformers import AutoTokenizer, EsmForProteinFolding

tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")

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
Hence, the first possibility to make the model lighter is to convert the model weights to `float16`, a sort of simple post-training quantization scheme. This is done with a single line of code

```python
model.esm = model.esm.half()
```
With this optimization of the model performance and requirements the memory usage is significantily decreased to 10.2GB, with little or no variation in the inference accuracy (more about this in the following).


## Quantization
Quantization is a powerful tool when it comes to achieving lower GPU memory requirements to run deep learning models. In particular, post training quantization (PTQ) aims at reducing the necessary resources for inference converting the wheights and the activations of the model to another type, e.g. `float8` or `int8`, while in principle preserving the model accuracy. To this end, it is possibile to use the python quantization toolkit `Quanto`([github](https://github.com/huggingface/quanto)), which is integrated in the Hugging Face `transformers` library and it is straightforward to implement. 

```python
from transformers import AutoTokenizer, EsmForProteinFolding, QuantoConfig

tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
quantization_config = QuantoConfig(weights="int8")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", device_map="cuda", quantization_config=quantization_config)

```
This code set the model weights (not the activations) to `int8`, hence reducing the memory prerequisite. Note that you might need to upgrade the `transformers` library in order 
to import `QuantoConfig`. This can be done with

```python
pip install --upgrade transformers

```

Comparing again the memory usage during the inference for the same protein results in only 6GB occupied!

## Analysis
Although these methods seem promising, some further analysis is necessary to evaluate their differences. In particular, what is still unclear is: 
- How does the memory requirement change for increasing sequence length? 
- How does the inference time correlates with the sequence length? Is there a speed up or a slow down for the quantized models? 
- How much accuracy loss the quantized models have?

#### 1. Memory
The following graph displays the GPU memory for the three different models considered, for an increasing sequence length.

![alt text](https://github.com/davideaguglia/ESMFold/blob/4f4412abbbe17a382c7d382acea32f4956c6a9a9/plots/memory.png)
In general, the models show the same correlation with the sequence length and the memory reduction between them is approximately constant: ~ 6 GB from the full model to the `float16` one and another ~ 4GB to the `int8` model. 

#### 2. Time
The second analysis that can be carried out is about the inference time. In particular, it is not obvious if the quantized models will also be faster, as matrix multiplication with `float32` are optimized. 
![alt text](https://github.com/davideaguglia/ESMFold/blob/b40fff68bd9a7be50e8706619eda7fade258a46d/plots/time.png)
From the graph it is clear how the inference time is approximately the same for the three models. However, for shorter sequences the int8 model appears to be the slowest one, probabily due to the lack of optimized kernels (as mentioned also [here](https://github.com/huggingface/quanto/blob/main/README.md)).

#### 3. Accuracy
Finally, one last important thing to study is the models' accuracy. We expect a loss in performance for the quantized models, but how much is this reduction?
To carry out this analysis it is possible to consider two measures of the model accuracy. The first and simplest one is the average pLDDT, which is an internal per-residue estimate of the model confidence on a scale from 0 to 100 (where an higher pLDDT is better). These values can be obtained from the `output`
```python
plddt = output['plddt'][0, :, 1]

```

Another possible method to measure the accuracy is through the contact map prediction. Given the inferred three-dimensional protein structure, which is given by the `positions`, it is possible to determine this matrix, which is nothing but the distance between all possible residue pairs. Then, this map can be compared to the one obtained from the experimentally measured proteins that can be found in the PDB.

![alt text](https://github.com/davideaguglia/ESMFold/blob/ef0ad408b26dee7d15755805e21ac5e3a6329a03/plots/acc.png)

This graph demonstrates how the performance loss due to the quantization procedure is minimal for both the configurations. The `float16` model performs essentially as the full model, while the huge reduction in GPU memory of the `int8` quantization determines a little more accuracy reduction.
To conclude this analysis the following graphs display, for each model, the correlations between the pLDDT the accuracy and the sequence length.
![alt text](https://github.com/davideaguglia/ESMFold/blob/0f91aea9c07d44897b11eb094b268c319e0f2dde/plots/models.png)


## Conclusions
This work demonstrates how it is possible to make use of available libraries, such as `transformers` and `quanto`, to easily implement quantization techniques to ESMFold. As a result of the analysis, this methods should be primarly intended for reducing the memory requirements necessary to run this model and not for increasing the inference time. It is remarkable that the benefits here outlined comes at little cost of performance, hence allowing for the use of this model for a precise protein structure prediction.
