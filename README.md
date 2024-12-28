# DCCF: Disentangled Contrastive Collaborative Filtering
SYSU ML course task: Replicating the DCCF model on the ReChorus framework


This repository contains the implementation of **DCCF: Disentangled Contrastive Collaborative Filtering**, 
a model designed to improve recommendation performance 
by adapting **graph contrastive learning** to **intent disentanglement** with self-supervision noise for 
collaborative filtering. 
The model has been successfully reproduced 
within the **Rechorus** framework, 
which provides a flexible environment for evaluating various recommendation models.

You can find the DCCF repository here: [DCCF Repository](https://github.com/HKUDS/DCCF).
Check out the framework here: [ReChorus Repository](https://github.com/THUwangcy/ReChorus).
## Our Modifications

In our implementation of the **DCCF** model, we made several important modifications to the original codebase:

1. We **added** the `DCCFRunner` and files replicated from the design of the original work.

2. The core implementation of the **DCCF** model is located in the `ReChrous/src/models/mymodel/DCCF.py` file, 
where the primary model architecture and logic are defined.



These changes were made to better align with the original framework, 
maintaining the overall structure and completeness of the **DCCF** model within the **Rechorus** framework.



## Getting Started

To get started with **DCCF** in the **Rechorus** framework, follow these steps:

1. Download or Clone this repository:
   ```bash
   git clone https://github.com/nbawwe/DCCF-ReChorus.git
   ```
2. Install the required dependencies:
  ```bash
    cd ReChorus
    pip install -r requirements.txt
  ```
3. Prepare the datasets

4. Train and evaluate the model:
  ```bash
    cd ReChorus
    cd src
    python main.py --model_name DCCF --emb_size 32 --lr 1e-4 --l2 1e-6 --path your_path_of_data_dir --dataset your_dataset --batch_size 2560
  ```

5. You can test the slicing of the Amazon Grocery dataset from the Rechorus framework in the DCCF folder using the original code.