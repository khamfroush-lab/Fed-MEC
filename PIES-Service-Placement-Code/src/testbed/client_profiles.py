# -- coding: future_fstrings --
DEFAULT_PROFILE, HIGH_ACCURACY, LOW_DELAY, HIGH_ACCURACY_LOW_DELAY = range(4)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import numpy.random as rand
    import seaborn as sns
    
    sns.set_style("ticks")
    
    """Just plot the distributions of the accuracy and delay generative functions given 
       the provided parameters. This is just to provide some easy analysis of what makes
       sense. We may need this to fine-tune the parameters for the real-world case given
       data about what the real-world delay looks like.
    """
    data = {
        "vals": [],
        "prof": [],
    }
    num_items = 100000

    ## High accuracy profile.
    vals = rand.normal(loc=0.75, scale=0.0625, size=num_items)
    vals = np.clip(vals, 0.0, 1.0)
    data["vals"].extend(vals)
    data["prof"].extend(["high_acc"] * num_items)

    ## Standard accuracy profile.
    vals = rand.normal(loc=0.5, scale=0.09375, size=num_items)
    vals = np.clip(vals, 0.0, 1.0)
    data["vals"].extend(vals)
    data["prof"].extend(["stand_acc"] * num_items)


    ## Low delay profile.
    vals = rand.normal(loc=1.0, scale=0.25, size=num_items)
    vals = np.clip(vals, 0.0, 5.0)
    data["vals"].extend(vals)
    data["prof"].extend(["low_del"] * num_items)

    ## Standard accuracy profile.
    vals = rand.normal(loc=3.0, scale=.75, size=num_items)
    vals = np.clip(vals, 0.0, 5.0)
    data["vals"].extend(vals)
    data["prof"].extend(["stand_del"] * num_items)

    sns.displot(x="vals", col="prof", col_wrap=2, hue="prof", data=data)
    plt.show()