# Data science techniques for condensed matter physics
This is the repository for the graduate course on "Data science techniques for condensed matter physics" at the Catholic University of the Sacred Hearth, Brescia, Italy (April 2022).


# Tutorial requirements 
We will use the tutorials from [Dive into Deep Learning (d2l.ai)](http://d2l.ai/index.html), an interactive deep learning book with code, math, and discussions, and it is adopted at 300 universities from 55 countries.    
We will use Python, and the PyTorch tutorials.

Thanks to [d2l.ai](http://d2l.ai/index.html) infrastructure, you will be able to run the tutorials simply using your browser.    

However, for the last tutorial on materials science data, you will need to run the Jupyter Notebook in your machine.
To this end, we suggest to install [conda](https://docs.conda.io/projects/conda/en/latest/index.html).
Moreover, with conda installed on your machine, you will be able to run locally all the d2l.ai tutorials.

### Conda installation
If you want to run the tutorials on your machine, please install conda.
You can find the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

## Tutorials from d2l.ai (no installation requireed)
1. Linear regression   
    - Linear regression (from scratch): [Webpage](http://d2l.ai/chapter_linear-networks/linear-regression-scratch.html) - [PyTorch](https://colab.research.google.com/github/d2l-ai/d2l-pytorch-colab/blob/master/chapter_linear-networks/linear-regression-scratch.ipynb)
    - Linear regression (concise implementation): [Webpage](http://d2l.ai/chapter_linear-networks/linear-regression-concise.html) - [PyTorch](https://colab.research.google.com/github/d2l-ai/d2l-pytorch-colab/blob/master/chapter_linear-networks/linear-regression-concise.ipynb)
    - Linear models for materials science ([NOMAD](https://www.nomad-coe.eu/) laboratory): [link](https://nomad-lab.eu/prod/analytics/public/user/40ae57ab-c7fe-4ba1-b7b5-a11de97d262b/notebooks/tutorials/compressed_sensing.ipynb)

2. Multi-layer perceptrons
    - Multilayer Perceptrons (from scratch): [Webpage](http://d2l.ai/chapter_multilayer-perceptrons/mlp-scratch.html) - [PyTorch](https://colab.research.google.com/github/d2l-ai/d2l-pytorch-colab/blob/master/chapter_multilayer-perceptrons/mlp-scratch.ipynb)
    - Multilayer Perceptrons (concise implementation): [Webpage](http://d2l.ai/chapter_multilayer-perceptrons/mlp-concise.html) - [PyTorch](https://colab.research.google.com/github/d2l-ai/d2l-pytorch-colab/blob/master/chapter_multilayer-perceptrons/mlp-concise.ipynb)
    - Multilayer perceptrons: predicting formation energies ([NOMAD](https://www.nomad-coe.eu/) laboratory): [link](https://nomad-lab.eu/prod/analytics/public/user/6cb97a6b-e14a-4559-ad0f-a3dde4a904ac/notebooks/tutorials/nn_regression.ipynb)

3. Convolutional Neural Networks (CNNs)
    - Convolutional Neural Networks (LeNet): [Webpage](http://d2l.ai/chapter_convolutional-neural-networks/lenet.html) - [PyTorch](https://colab.research.google.com/github/d2l-ai/d2l-pytorch-colab/blob/master/chapter_convolutional-neural-networks/lenet.ipynb)
    - Modern Convolutional Neural Networks (AlexNet): [Webpage](http://d2l.ai/chapter_convolutional-modern/alexnet.html) - [PyTorch](https://colab.research.google.com/github/d2l-ai/d2l-pytorch-colab/blob/master/chapter_convolutional-modern/alexnet.ipynb)

4. Generative Adversarial Networks (GANs)
    - GANs with multi-layer perceptrons: [Webpage](http://d2l.ai/chapter_generative-adversarial-networks/gan.html) - [PyTorch](https://colab.research.google.com/github/d2l-ai/d2l-pytorch-colab/blob/master/chapter_generative-adversarial-networks/gan.ipynb)
    - Deep Convolutional Adversarial Networks: [Webpage](http://d2l.ai/chapter_generative-adversarial-networks/dcgan.html) - [PyTorch](https://colab.research.google.com/github/d2l-ai/d2l-pytorch-colab/blob/master/chapter_generative-adversarial-networks/dcgan.ipynb)

5. Recurrent Neural Networks (RNNs)
    - Recurrent neural networks (from scratch): [Webpage](http://d2l.ai/chapter_recurrent-neural-networks/rnn-scratch.html) - [PyTorch](https://colab.research.google.com/github/d2l-ai/d2l-pytorch-colab/blob/master/chapter_recurrent-neural-networks/rnn-scratch.ipynb)
    - Recurrent neural networks (concise implementation): [Webpage](http://d2l.ai/chapter_recurrent-neural-networks/rnn-concise.html) - [PyTorch](https://colab.research.google.com/github/d2l-ai/d2l-pytorch-colab/blob/master/chapter_recurrent-neural-networks/rnn-concise.ipynb)
    - Long short term memory (LSTM): [Webpage](http://d2l.ai/chapter_recurrent-modern/lstm.html) - [PyTorch](https://colab.research.google.com/github/d2l-ai/d2l-pytorch-colab/blob/master/chapter_recurrent-modern/lstm.ipynb)


## Tutorial "Unsupervised learning on physics data"

This is an example of machine learning analysis on real physics data.    
The data was obtained by querying the NOMAD lab database [link](https://nomad-lab.eu/prod/rae/gui/search) and search for 2D systems, single-point calculation (see [here](https://nomad-lab.eu/prod/rae/gui/search?results=entries&visualization=properties&dft.system=2D&dft.workflow.workflow_type=single_point)). 
Only a subset of results were retrieved to have less than 250Mb.    
Feel free to experiment with other queries, since the code is transferable.

To execute the tutorial, you have two choices:   

- Run the Jupyter Notebook provided in this Github using [Colab](https://colab.research.google.com/)   
- Run the Jupyter Notebook provided on your local machine. This is the most demanding way, but it is closer to a real-world data scientist setup.

### A. Colab (requires Google Account, including GoogleDrive)

1. Download the zip file containing the data from [here](https://onedrive.live.com/?id=c78a18fba37b17ea%2152841&cid=C78A18FBA37B17EA)
2. Extract the zip file.
3. Upload the folder to your Google Drive
4. Launch the Jupyter Notebook:

    jupyter noteboook *name_of_the_notebook*

5. In the Jupyter Notebook, set the *data_folder* to the folder where you put the data. If you did not change the folder name, the data folder in Google Drive is: *"/content/gdrive/My Drive/2d_materials_nomad"*
6. Run the cells in the notebook

### B. Local machine

The first step is to perform a local installation. After having installed conda (as explained above), you need to create a virtual environment ([documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)).   


    # clone this repository
    git clone https://github.com/angeloziletti/unicatt-ds-physics
    # go in the main folder
    cd data-science-physics

    # create a virtual environment and activate it
    conda create -n ds-unicatt python=3.7
    conda activate ds-unicatt

    # install the required packages
    pip install -r requirements.txt


After you have performed the local installation:

1. Download the zip file containing the data from the link provided during the class.
2. Extract the zip file.
3. Launch the Jupyter Notebook:

    jupyter noteboook *name_of_the_notebook*

4. In the Jupyter Notebook, set the *data_folder* to the folder where you put the data. 
5. Run the cells in the Jupyter notebook






