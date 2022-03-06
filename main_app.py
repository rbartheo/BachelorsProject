import tkinter.messagebox
from tkinter import *
import tkinter as tk
from tkinter import ttk
from load_data import load_dataset1, load_dataset2, load_dataset4, load_dataset5
from plot_data import plot_series
from PIL import ImageTk, Image
from clustering_algorithms import k_means, affinity, agglomerative, birch, dbscan, mini_kmeans, spectral
from data_descriptions import load_data_description

# Create a main window
root = tk.Tk()
# Set a resolution
root.geometry('1280x720')
# Can't resize window
root.resizable(False, False)
# Window always open maximized
root.state("zoomed")
# Set title of the window
root.title("Time series clustering")


def clear_tree():
    """
    Function that clear treeview widget
    """
    data_treeview.delete(*data_treeview.get_children())


def callbackFunc(event):
    """
    Function used in combobox. At the beginning it clears data, next checks for selected dataset in combobox and
    finally loads data into treeview widget. It also calls plot_series function which makes a plot of a laoded series
    ,load_image function which loads plot image into label widget and load_data_desc function which loads
    data description from .txt file and put it in textbox.
    :param event: combobox selected item
    :type event: event
    """
    clear_tree()
    if data_combobox.get() == 'dataset 1':
        root.mySeries, root.nameOfSeries, root.mySeriesIndex = load_dataset1()
        df = root.mySeriesIndex[1]
        data_treeview['column'] = list(df.columns)
        data_treeview["show"] = "headings"
        for column in data_treeview["column"]:
            data_treeview.heading(column, text=column)
        data_list = df.to_numpy().tolist()
        for data in data_list:
            data_treeview.insert("", "end", values=data)
        plot_series(root.mySeries, root.nameOfSeries)
        load_image()
        load_data_desc("dataset1_description")
        plot_buttons()
    if data_combobox.get() == 'dataset 2':
        root.mySeries, root.nameOfSeries, mySeriesIndex = load_dataset2()
        df = mySeriesIndex[1]
        data_treeview['column'] = list(df.columns)
        data_treeview["show"] = "headings"
        for column in data_treeview["column"]:
            data_treeview.heading(column, text=column)
        data_list = df.to_numpy().tolist()
        for data in data_list:
            data_treeview.insert("", "end", values=data)
        plot_series(root.mySeries, root.nameOfSeries)
        load_image()
        load_data_desc("dataset2_description")
        plot_buttons()
    if data_combobox.get() == 'dataset 3':
        root.mySeries, root.nameOfSeries, mySeriesIndex = load_dataset5()
        df = mySeriesIndex[1]
        data_treeview['column'] = list(df.columns)
        data_treeview["show"] = "headings"
        for column in data_treeview["column"]:
            data_treeview.heading(column, text=column)
        data_list = df.to_numpy().tolist()
        for data in data_list:
            data_treeview.insert("", "end", values=data)
        plot_series(root.mySeries, root.nameOfSeries)
        load_image()
        load_data_desc("dataset3_description")
        plot_buttons()
    if data_combobox.get() == 'dataset 4':
        root.mySeries, root.nameOfSeries, mySeriesIndex = load_dataset4()
        df = mySeriesIndex[1]
        data_treeview['column'] = list(df.columns)
        data_treeview["show"] = "headings"
        for column in data_treeview["column"]:
            data_treeview.heading(column, text=column)
        data_list = df.to_numpy().tolist()
        for data in data_list:
            data_treeview.insert("", "end", values=data)
        plot_series(root.mySeries, root.nameOfSeries)
        load_image()
        load_data_desc("dataset4_description")
        plot_buttons()


def load_data_desc(dataset_name):
    """
    A function that sets textbox state to write then loads a content of data_description text file and then sets
    textbox state to read only
    :param dataset_name: name of .txt data set description file
    :type dataset_name: str
    """
    data_description_textbox.config(state="normal")
    data_description_textbox.delete('1.0', "end")
    data_description = load_data_description(dataset_name)
    data_description_textbox.insert("end", data_description)
    data_description_textbox.config(state="disabled")


def load_image():
    """
    A function that opens saved plot image of a data series, resizes it and loads it into image label
    """
    img = Image.open("Plots/whole_series_plot.png")
    new_width = 1800
    new_height = 800
    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    img.save('Plots/whole_series_plot_resized.png')
    data_image = ImageTk.PhotoImage(
        Image.open("Plots/whole_series_plot_resized.png"))
    image_label.configure(image=data_image)
    image_label.image = data_image


# Create two global lists to save data about series
root.mySeries = []
root.nameOfSeries = []


def k_means_button():
    """
    A function that handles an event of K_Means button. It checks if two global lists are empty, if the lists are empty
    it shows error message, if the lists contain data it calls k_means function, which does k_means clustering and calls
    create_window function which shows new window with data about k_means clustering
    :return: None if lists are empty
    :rtype: None
    """
    if len(root.mySeries) == 0 or len(root.nameOfSeries) == 0:
        tkinter.messagebox.showerror(title="Error", message="Select dataset.")
        return
    k_means(root.mySeries, root.nameOfSeries)
    distribution, score = k_means(root.mySeries, root.nameOfSeries)
    create_window(distribution, score, "Kmeans")


def affinity_button():
    """
    A function that handles an event of Affinity Propagation button. It checks if two global lists are empty,
    if the lists are empty it shows error message, if the lists contain data it calls som function, which does affinity
    propagation clustering and calls create_window function which shows new window with data about Affinity propagation
    clustering
    :return: None if lists are empty
    :rtype: None
    """
    if len(root.mySeries) == 0 or len(root.nameOfSeries) == 0:
        tkinter.messagebox.showerror(title="Error", message="Select dataset.")
        return None
    affinity(root.mySeries, root.nameOfSeries)
    distribution, score = affinity(root.mySeries, root.nameOfSeries)
    create_window(distribution, score, "Affinity")


def agglomerative_button():
    """
    A function that handles an event of agglomerative button. It checks if two global lists are empty, if the lists are
    empty it shows error message, if the lists contain data it calls agglomerative function, which does
    Agglomerative clustering and calls create_window function which shows new window with data about
    agglomerative clustering
    :return: None if lists are empty
    :rtype: None
    """
    if len(root.mySeries) == 0 or len(root.nameOfSeries) == 0:
        tkinter.messagebox.showerror(title="Error", message="Select dataset.")
        return None
    agglomerative(root.mySeries, root.nameOfSeries)
    distribution, score = agglomerative(root.mySeries, root.nameOfSeries)
    create_window(distribution, score, "Agglomerative")


def birch_button():
    """
    A function that handles an event of birch button. It checks if two global lists are empty, if the lists are
    empty it shows error message, if the lists contain data it calls birch function, which does BIRCH clustering
    and calls create_window function which shows new window with data about BIRCH clustering
    :return: None if lists are empty
    :rtype: None
    """
    if len(root.mySeries) == 0 or len(root.nameOfSeries) == 0:
        tkinter.messagebox.showerror(title="Error", message="Select dataset.")
        return None
    birch(root.mySeries, root.nameOfSeries)
    distribution, score = birch(root.mySeries, root.nameOfSeries)
    create_window(distribution, score, "Birch")


def dbscan_button():
    """
    A function that handles an event of dbscan button. It checks if two global lists are empty, if the lists are
    empty it shows error message, if the lists contain data it calls dbscan function, which does DBSCAN clustering
    and calls create_window function which shows new window with data about DBSCAN clustering
    :return: None if lists are empty
    :rtype: None
    """
    if len(root.mySeries) == 0 or len(root.nameOfSeries) == 0:
        tkinter.messagebox.showerror(title="Error", message="Select dataset.")
        return None
    if data_combobox.get() == 'dataset 1':
        dbscan(root.mySeries, root.nameOfSeries, 3, 2)
        distribution, score = dbscan(root.mySeries, root.nameOfSeries, 3, 2)
    elif data_combobox.get() == 'dataset 2':
        dbscan(root.mySeries, root.nameOfSeries, 3, 2)
        distribution, score = dbscan(root.mySeries, root.nameOfSeries, 3, 2)
    elif data_combobox.get() == 'dataset 3':
        dbscan(root.mySeries, root.nameOfSeries, 0.4, 3)
        distribution, score = dbscan(root.mySeries, root.nameOfSeries, 0.4, 3)
    elif data_combobox.get() == 'dataset 4':
        dbscan(root.mySeries, root.nameOfSeries, 0.64, 2)
        distribution, score = dbscan(root.mySeries, root.nameOfSeries, 0.64, 2)
    create_window(distribution, score, "Dbscan")


def minikmeans_button():
    """
    A function that handles an event of minikmeans button. It checks if two global lists are empty, if the lists are
    empty it shows error message, if the lists contain data it calls minikmeans function, which does Mini batch K-Means
    clustering and calls create_window function which shows new window with data about Mini batch K-Means clustering
    :return: None if lists are empty
    :rtype: None
    """
    if len(root.mySeries) == 0 or len(root.nameOfSeries) == 0:
        tkinter.messagebox.showerror(title="Error", message="Select dataset.")
        return None
    mini_kmeans(root.mySeries, root.nameOfSeries)
    distribution, score = mini_kmeans(root.mySeries, root.nameOfSeries)
    create_window(distribution, score, "Minikmeans")


def spectral_button():
    """
    A function that handles an event of spectral button. It checks if two global lists are empty, if the lists are
    empty it shows error message, if the lists contain data it calls spectral function, which does Spectral clustering
    and calls create_window function which shows new window with data about Spectral clustering
    :return: None if lists are empty
    :rtype: None
    """
    if len(root.mySeries) == 0 or len(root.nameOfSeries) == 0:
        tkinter.messagebox.showerror(title="Error", message="Select dataset.")
        return None
    spectral(root.mySeries, root.nameOfSeries)
    distribution, score = spectral(root.mySeries, root.nameOfSeries)
    create_window(distribution, score, "Spectral")


def create_window(distribution, scores, method):
    """
    A function that crates a new window after clicking one of seven buttons. First it creates a new window with
    parameters, next it creates two textbox widgets which contains data from clustering results (score and cluster
    mapping). Window also contains label for the clustering results plots and two button to navigate between plots
    :param method: A string of the method used to cluster data
    :type method: str
    :param distribution: Distribution of which data is in which cluster
    :type distribution: list
    :param scores: cluster evaluation metrics results
    :type scores: list
    """
    method = method
    curr_image = -1

    def plot_dataseries(method, plot_type):
        """
        A function that makes, resizes, saves data plot and loads it to a image label
        :param method: Name of method used to cluster data
        :type method: str
        :param plot_type: Name of the plot type (barplot, scatterplot, clusterplot etc.)
        :type plot_type: str
        """
        img = Image.open(f"Plots/{method}/{method.lower()}{plot_type}")
        new_width = 1800
        new_height = 840
        img = img.resize((new_width, new_height), Image.ANTIALIAS)
        img.save(f"Plots/{method}/{method.lower()}{plot_type}")
        data_image = ImageTk.PhotoImage(
            Image.open(f"Plots/{method}/{method.lower()}{plot_type}"))
        plot_label.configure(image=data_image)
        plot_label.image = data_image

    def back_button():
        """
        An event function for the button that is used to go back to the previous data plot.
        :return: None if curr_image = 1
        :rtype: None
        """
        nonlocal curr_image
        bar = "_barplot.png"
        scatter = "_scatterplot.png"
        clusterplot = "_clusteringplot.png"
        if curr_image == 1:
            return
        elif curr_image == 2:
            plot_dataseries(method, clusterplot)
            curr_image = 1
        elif curr_image == 3:
            plot_dataseries(method, scatter)
            curr_image = 2
        elif curr_image == 4:
            plot_dataseries(method, bar)
            curr_image = 3

    def forward_button():
        """
        An event function for the button that is used to go forward to the next data plot.
        """
        nonlocal curr_image
        bar = "_barplot.png"
        scatter = "_scatterplot.png"
        hierarchical = "_hierarchicalplot.png"
        varplot = "_varplot.png"

        if curr_image == 1:
            plot_dataseries(method, scatter)
            curr_image = 2
        elif curr_image == 2:
            plot_dataseries(method, bar)
            curr_image = 3
        elif curr_image == 3 and method == "Agglomerative":
            plot_dataseries(method, hierarchical)
            curr_image = 4
        elif curr_image == 3 and method == "Dbscan":
            plot_dataseries(method, varplot)
            curr_image = 4

    # Create new window after button click
    clustering_results = Toplevel(root)
    # Set title of the window
    clustering_results.title("Clustering results")
    # Can't resize the window
    clustering_results.resizable(False, False)
    # Window will always open maximized
    clustering_results.state("zoomed")

    # Create a textbox
    scores_textbox = Text(clustering_results)
    # Place a textbox in the window
    scores_textbox.place(relx=0.511, rely=0.038, relheight=0.202, relwidth=0.24)
    # Insert data into textbox
    scores_textbox.insert('end', "Silhouette score: ")
    scores_textbox.insert('end', round(scores[0], 3))
    scores_textbox.insert('end', "\n")
    scores_textbox.insert('end', "Harabasz-Caliński score: ")
    scores_textbox.insert('end', round(scores[1], 3))
    scores_textbox.insert('end', "\n")
    scores_textbox.insert('end', "Davies-Bouldin score: ")
    scores_textbox.insert('end', round(scores[2], 3))
    # Set textbox status to read only
    scores_textbox.config(state='disabled')

    # Create a textbox
    cluster_mapping_textbox = Text(clustering_results)
    # Place a textbox in the window
    cluster_mapping_textbox.place(relx=0.226, rely=0.038, relheight=0.202, relwidth=0.249)
    # Insert data into textbox
    cluster_mapping_textbox.insert("end", distribution)
    # Set textbox status to read only
    cluster_mapping_textbox.config(state='disabled')

    # Create a label
    plot_label = tk.Label(clustering_results)
    # Place label in window
    plot_label.place(relx=0.081, rely=0.250, height=840, width=1554)

    # string variable used in if statements
    clustering = "_clusteringplot.png"

    if method == "Agglomerative":
        plot_dataseries(method, clustering)
        curr_image = 1
    elif method == "Birch":
        plot_dataseries(method, clustering)
        curr_image = 1
    elif method == "Kmeans":
        plot_dataseries(method, clustering)
        curr_image = 1
    elif method == "Dbscan":
        plot_dataseries(method, clustering)
        curr_image = 1
    elif method == "Minikmeans":
        plot_dataseries(method, clustering)
        curr_image = 1
    elif method == "Affinity":
        plot_dataseries(method, clustering)
        curr_image = 1
    elif method == "Spectral":
        plot_dataseries(method, clustering)
        curr_image = 1

    # Create label
    name_label1 = tk.Label(clustering_results)
    name_label1.place(relx=0.231, rely=0.009, height=27, width=446)
    name_label1.configure(text="Cluster mapping")

    # Create label
    name_label2 = tk.Label(clustering_results)
    name_label2.place(relx=0.517, rely=0.009, height=27, width=423)
    name_label2.configure(text="Clustering efficiency measure results")

    # Create button
    back_button = Button(clustering_results, text="<", command=back_button, bg="darkgray")
    back_button.place(relx=0.016, rely=0.558, height=54, width=57)

    # Create button
    forward_button = Button(clustering_results, text=">", command=forward_button, bg="darkgray")
    forward_button.place(relx=0.947, rely=0.558, height=54, width=57)


# Create a combobox widget
selected_data = tk.StringVar()
data_combobox = ttk.Combobox(root, textvariable=selected_data)
# Insert values into combobox
data_combobox['values'] = ('dataset 1',
                           'dataset 2',
                           'dataset 3',
                           'dataset 4')
# Cant change values of a combobox
data_combobox['state'] = 'readonly'

# Place combobox in main window
data_combobox.place(relx=0.048, rely=0.322, relheight=0.029, relwidth=0.088)

# Create treeview widget
data_treeview = ttk.Treeview(root)

# Place treeview in main window
data_treeview.place(relx=0.048, rely=0.044, relheight=0.219, relwidth=0.511)
# Create a vertical scrollbar for treeview
data_vertical_scroll = ttk.Scrollbar(root, orient="vertical", command=data_treeview.yview)
# Place vertical scroll in the main window
data_vertical_scroll.place(relx=0.559, rely=0.047, relheight=0.219, relwidth=0.010)
# Create a horizontal scrollbar for treeview
data_horizontal_scroll = ttk.Scrollbar(root, orient="horizontal", command=data_treeview.xview)
# Place horizontal scroll in the main window
data_horizontal_scroll.place(relx=0.047, rely=0.265, height=18, width=980)
# Set vertical scrollbar to scroll treeview content
data_treeview.configure(yscrollcommand=data_vertical_scroll.set, xscrollcommand=data_horizontal_scroll.set)

# Bind selected item from combobox and call function to load data to selected widgets
data_combobox.bind("<<ComboboxSelected>>", callbackFunc)

# Create textbox
data_description_textbox = Text(root)
# Place textbox in main window
data_description_textbox.place(relx=0.587, rely=0.047, relheight=0.219, relwidth=0.355)
# Create vertical scrollbar for the textbox
description_vertical_scrollbar = ttk.Scrollbar(root, orient="vertical", command=data_description_textbox.yview)
# place vertical scrollbar in the main window
description_vertical_scrollbar.place(relx=0.942, rely=0.047, relheight=0.219, relwidth=0.008)
# Set vertical scrollbar to scroll textbox content
data_description_textbox.configure(yscrollcommand=description_vertical_scrollbar.set)

# Create label
image_label = tk.Label(root)
image_label.place(relx=0.151, rely=0.290, height=800, width=1523)

# Create button
affinity_button = Button(text="Affinity propagation", command=affinity_button, bg="darkgray")
affinity_button.place(relx=0.050, rely=0.456, height=34, width=145)

# Create button
kmeans_button = Button(text="K-means", command=k_means_button, bg="darkgray")
kmeans_button.place(relx=0.050, rely=0.512, height=34, width=145)

# Create button
birch_button = Button(text="BIRCH", command=birch_button, bg="darkgray")
birch_button.place(relx=0.050, rely=0.569, height=34, width=145)

# Create button
dbscan_button = Button(text="DBSCAN", command=dbscan_button, bg="darkgray")
dbscan_button.place(relx=0.050, rely=0.624, height=34, width=145)

# Create button
spectral_button = Button(text="Spectral clustering", command=spectral_button, bg="darkgray")
spectral_button.place(relx=0.050, rely=0.683, height=34, width=145)

# Create button
agglomerative_button = Button(text="Agglomerative clustering", command=agglomerative_button, bg="darkgray")
agglomerative_button.place(relx=0.050, rely=0.74, height=34, width=145)

# Create button
minikmeans_button = Button(text="Mini batch K-Means", command=minikmeans_button, bg="darkgray")
minikmeans_button.place(relx=0.050, rely=0.796, height=34, width=145)

# Create label
buttons_label = Label(root)
# Place label in main window
buttons_label.place(relx=0.0435, rely=0.415, height=15, width=164)
# Set the text of the label
buttons_label.configure(text="Clustering algorithms")

# Create label
data_label = Label(root)
# Place label in main window
data_label.place(relx=0.043, rely=0.287, height=20, width=164)
# Set the text of the label
data_label.configure(text="Datasets")

# Create label
data_tree_label = Label(root)
# Place label in main window
data_tree_label.place(relx=0.059, rely=0.009, height=29, width=932)
# Set the text of the label
data_tree_label.configure(text="CSV file data")

# Create label
data_description_label = Label(root)
# Place label in main window
data_description_label.place(relx=0.610, rely=0.009, height=30, width=603)
# Set the text of the label
data_description_label.configure(text="Data set description")


def load_image2(data, plot):
    """
    A function that loads file from WholePlots folder, resizes it and loads resized image into image_label
    :param data: Name of folder with plot
    :type data: str
    :param plot: End of plot name _x, _y, _z
    :type plot: str
    """
    img = Image.open(f"Plots/WholePlots/{data}/whole_series_plot_{plot}.png")
    new_width = 1800
    new_height = 800
    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    img.save(f"Plots/WholePlots/{data}/whole_series_plot_{plot}.png")
    data_image = ImageTk.PhotoImage(
        Image.open(f"Plots/WholePlots/{data}/whole_series_plot_{plot}.png"))
    image_label.configure(image=data_image)
    image_label.image = data_image


def plot_buttons():
    """
    A function that when data set is loaded makes two buttons to switch between plots
    """
    image_number = 1

    def next_plot():
        """
        A function that is used as an event in main_plot_forward_button. It shows a plot that is assigned to
        image_number variable. It checks which dataset is loaded ant then it loads plot with load_image2 function
        """
        nonlocal image_number
        if data_combobox.get() == 'dataset 1':
            if image_number == 1:
                load_image2("Data1", "y")
                image_number = 2
            elif image_number == 2:
                load_image2("Data1", "z")
                image_number = 2
        if data_combobox.get() == 'dataset 2':
            if image_number == 1:
                load_image2("Data2", "x")
                image_number = 2
            elif image_number == 2:
                load_image2("Data2", "z")
                image_number = 2
        if data_combobox.get() == 'dataset 3':
            if image_number == 1:
                load_image2("Data3", "x")
                image_number = 2
            elif image_number == 2:
                load_image2("Data3", "y")
                image_number = 3
            elif image_number == 3:
                load_image2("Data3", "z")
                image_number = 3
        if data_combobox.get() == 'dataset 4':
            if image_number == 1:
                load_image2("Data4", "x")
                image_number = 2
            elif image_number == 2:
                load_image2("Data4", "y")
                image_number = 3
            elif image_number == 3:
                load_image2("Data4", "z")
                image_number = 3

    def previous_plot():
        """
        A function which is used as an event for main_plot_back button. It checks for loaded dataset and then
        it calls function that loads an image into label by calling function load_image2
        """
        nonlocal image_number
        if data_combobox.get() == 'dataset 1':
            if image_number == 1:
                load_image()
                image_number = 1
            elif image_number == 2:
                load_image2("Data1", "y")
                image_number = 1
        if data_combobox.get() == 'dataset 2':
            if image_number == 1:
                load_image()
                image_number = 1
            elif image_number == 2:
                load_image2("Data2", "x")
                image_number = 1
        if data_combobox.get() == 'dataset 3':
            if image_number == 1:
                load_image()
                image_number = 1
            elif image_number == 2:
                load_image2("Data3", "x")
                image_number = 1
            elif image_number == 3:
                load_image2("Data3", "y")
                image_number = 2
        if data_combobox.get() == 'dataset 4':
            if image_number == 1:
                load_image()
                image_number = 1
            elif image_number == 2:
                load_image2("Data4", "x")
                image_number = 1
            elif image_number == 3:
                load_image2("Data4", "y")
                image_number = 2

    # Create button
    main_plot_forward = tk.Button(root, command=next_plot)
    # Place button in main window
    main_plot_forward.place(relx=0.947, rely=0.485, height=64, width=67)
    # Set button text
    main_plot_forward.configure(text=">", bg="darkgray")

    # Create button
    main_plot_back = tk.Button(root, command=previous_plot)
    # Place button in main window
    main_plot_back.place(relx=0.947, rely=0.57, height=64, width=67)
    # Set button text
    main_plot_back.configure(text="<", bg="darkgray")

# Mainloop of the application
root.mainloop()