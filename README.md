# Electromagnetic shower generation with Graph Neural Networks

## Training
 
 To run training procedure type in terminal following command:

```
python training.py --datafile ./data/showers_all_energies_1_5.pkl  --max_prev_node 12 --embedding_size 196 --edge_rnn_embedding_size 16 
--embedding_size_gcn 4 --num_layers_gcn 3 --mixture_size 12 --lr 1e-4
--project_name shower_generation --work_space schattengenie
```

Weights of neural networks will be saved on disk each 10 epochs. 
`--max_prev_node 12 --embedding_size 196 --edge_rnn_embedding_size 16` are architecture parameters of GraphRNN, please refer to original [paper](https://arxiv.org/abs/1802.08773).

`--embedding_size_gcn 4 --num_layers_gcn 3 --mixture_size 12` define Graph Convolution architecture. 
`--embedding_size_gcn` corresponds for the dense layer size in [EdgeConv layer](https://arxiv.org/abs/1801.07829), `--num_layers_gcn 3` corresponds for number of EdgeConv layers used and `--mixture_size` define number of Gaussian distributions proposed by Mixture Density Network.

## Visualization

To generate EM-showers open in `Jupyter Notebook` `showers_generation_vizualizations.ipynb` file. Enter correct path to network weights and you will be able generate new showers.
