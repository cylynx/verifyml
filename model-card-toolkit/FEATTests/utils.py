from io import BytesIO
import matplotlib.pyplot as plt
import base64


def plot_to_str():
    # Utility function that will export a plot to a base-64 encoded string.
    img = BytesIO()
    plt.savefig(img, format="png")
    return base64.encodebytes(img.getvalue()).decode("utf-8")
