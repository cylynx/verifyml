import matplotlib.pyplot as plt
import tempfile
from fpdf import FPDF

fig = plt.figure()
fig.add_subplot(111)

with tempfile.TemporaryDirectory() as tmp:
    print(f"created temporary directory {tmp}")

    plt.figure(figsize=(5.2, 3), dpi=100)
    plt.plot([1, 2, 3, 4])
    plt.ylabel("some numbers")

    img_fp = f"{tmp}/test.png"

    plt.savefig(img_fp)

    pdf = FPDF()
    pdf.add_page()
    pdf.image(img_fp)

    pdf.output("/mnt/c/Users/Jason/Documents/test.pdf", "F")
