from data import generate_nonlinear_data
from visual import plot_2d_data

def main():
    x,y=generate_nonlinear_data()
    plot_2d_data(x,y,title="Non linear data")
if __name__=="__main__":
    main()