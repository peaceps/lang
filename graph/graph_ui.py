import io
import matplotlib
matplotlib.use("Qt5Agg") 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from langgraph.graph import StateGraph


def display(graph: StateGraph) -> None:    
    imgbytes = graph.get_graph(xray=True).draw_mermaid_png()
    img = mpimg.imread(io.BytesIO(imgbytes), format="png")
    _, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img)
    ax.axis("off")
    plt.tight_layout()
    plt.show()