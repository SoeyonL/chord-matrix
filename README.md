# chord-matrix

Collection of python functions to analyse correlation matrix and represent them using matrix or chord representation.

## Description

A little package or a notebook is available, the functions allow load, export, statistic computation and representation. It is recommended to create a virtual environment, the process is described in the notebook and below.

## Virtual environment

Since the `openchord` dependency is quite unstable, it's prefered to use python's virtual environment. You will have to open a terminal or a powershell.

Linux/Mac:

`python3 -m venv openchord # create the environment`
`source openchord/bin/activate # activate it`
`pip install ipykernel ipywidgets numpy pandas tables scipy matplotlib seaborn openchord # install the packages`
`python -m ipykernel install --user --name=openchord # install it for jupyter`

Windows:

`py -m venv openchord # create the environment`
`openchord\Scripts\activate # activate it`
`py -m pip install ipykernel ipywidgets numpy pandas tables scipy matplotlib seaborn openchord # install the packages`
`ipython kernel install --user --name=openchord # install it for jupyter`

This procedure suppose you have python already installed and ready to use, if you do not intend to use the notebook, skip the last command.

## Usage

We suppose the files are `csv`, modification can be done to support `excel` files, see [Pandas' documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html) for more.

One can load a single file with the function `load` or a directory containing multiples files with `load_directory`. It can be used as follow: 

```
csv = load("path/to/file.csv", ",")

# OR

csv_list = load_directory("path/to/directory/", ",")

### to show a matrix
display(csv)

# OR

display(csv[0])
```

Then using the result of the loading function, one can use the statistics function to work on them. One function can ask or either a single file or a list of files. Do not hesitate to read the description next to the function for more information!

```
mean_matrix = mean(csv_list)

std_matrix = std(csv_list_)
```

Then you can plot the result with a simple function:

```
plot(csv)

# With some arguments
plot(csv, title = "My title", save = "path/to/save", show = False, vmin = -0.25, vmax = 4)

# to show the significant P-value:

plot_p(csv)
```

And finally a function to show the correlation using a chord representation:

```
f = chord(csv)
f.show() # due to limitation in the library, to show them in the notebook, you have to do this
```

This function take a lot of arguments to tweak the plot, you can pass then when calling the `chord` function. This function is not suitable for negative values.

**But beware!** If NaN (Not a Number) value are in the matrix, the representation will fail!

## Dependencies

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `seaborn`
- `openchord`

## Citation

If you are using this little library, please link this page in your paper.