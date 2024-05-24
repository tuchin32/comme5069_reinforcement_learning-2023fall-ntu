import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    tile_count = {
        2: 1,
        16: 1,
        32: 3,
        64: 1,
        128: 16,
        256: 45,
        512: 29,
        1024: 4,
    }

    # x = np.array(list(tile_count.keys()))
    x = np.arange(0, len(tile_count))
    y = np.array(list(tile_count.values()))
    print(x)
    print(y)
    plt.bar(x, y)
    # Mark y values
    for i, v in enumerate(y):
        plt.text(i, v + 0.3, str(v), ha='center', va='bottom')
    plt.xticks(x, list(tile_count.keys()))
    plt.xlabel('Best Tile')
    plt.ylabel('Frequency')
    plt.ylim(0, 50)
    plt.title('Best Tile Distribution')
    plt.savefig('tile_histogram.png')