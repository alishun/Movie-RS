# Recommendation System Project
## Description
This project involves the development of a hybrid recommendation system using DCNv2 and evolutionary algorithms to optimize for accuracy, diversity, novelty, and user coverage. The models are implemented in Python and tested with Jupyter Notebook.

I have authored a research paper related to this project. If you are interested in reading it, please send me an email at [alisonluyin@gmail.com](mailto:alisonluyin@gmail.com) for access.

## Installation

To run this project, ensure you have Python 3.x installed.
1. Clone the repository and navigate into the project directory:

```bash
git clone https://github.com/yourusername/recommendation-system.git
cd recommendation-system
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install the required libraries:
```bash
pip install -r requirements.txt
```

## Running the Notebooks
Ensure Jupyter Notebook is installed.
1. Start the Jupyter notebook server:
```bash
jupyter notebook
```
2. Open and run the desired notebook from the Jupyter interface.

## Data
### [MovieLens 100k latest](https://grouplens.org/datasets/movielens/)
The MovieLens 100k Latest Dataset [1] was utilized in this project for building and evaluating recommendation models. This dataset contains user ratings and movie metadata, making it suitable for collaborative filtering and content-based recommendation systems.

[1] F. Maxwell Harper and Joseph A. Konstan. 2016. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872

### [IMDb Datasets](https://datasets.imdbws.com/)
IMDb datasets are also utilized in the project to supplement the data from MovieLens. Since the original IMDb datasets contain hundreds of millions of lines of information, they have already been preprocessed and combined with the MovieLens dataset. The result is stored in `datasets/ml-imdb.csv.py`.

## Acknowledgements
I would like to express my sincere gratitude to Sergio Maffeis, my project supervisor at Imperial College London, for his invaluable support and guidance throughout this project.
