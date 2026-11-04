import matplotlib.pyplot as plt

def plot_2d_data (x,y,title="Linear dats"):
    plt.scatter(x[:,0],x[:,1],c=y)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("y")
    plt.show()