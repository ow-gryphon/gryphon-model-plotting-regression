import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

color_dict = {
    "dark_blue": "#002C77",
    "blue": "#009DE0",
    "light_blue": "#76D3FF",
    "dark_gray": "#565656",
    "gray": "#949494",
    "light_gray": "#DADADA",
    "dark_green": "#275D38",
    "green": "#00AC41",
    "light_green": "#ADDFB3",
    "dark_teal": "#004C6C",
    "teal": "#0077A0",
    "light_teal": "#9CD9E4",
    "dark_turquoise": "#005E5D",
    "turquoise": "#00968F",
    "light_turquoise": "#98DBCE",
    "dark_yellow": "#965D00",
    "yellow": "#FFBE00",
    "light_yellow": "#FFE580",
    "dark_orange": "#A32E00",
    "orange": "#FF8C00",
    "light_orange": "#FFCA94",
    "dark_crimson": "#9A1C1F",
    "crimson": "#EF4E45",
    "light_crimson": "#FFAEA6",
    "dark_pink": "#B2025B",
    "pink": "#EE3D8B",
    "light_pink": "#F8ACBE",
    "dark_purple": "#463282",
    "purple": "#8246AF",
    "light_purple": "#CCB3E0",
    "dark_blue_gray": "#4E6287",
    "blue_gray": "#8096B2",
    "light_blue_gray": "#BED3E4",
    "black": "#000000",
}

def hex_to_rgb(value):
    """Return (red, green, blue) for the color given as #rrggbb."""
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def colormap_linear(color_hexes, name, N = 100):
    colors = []
    for color_hex in color_hexes:
        colors.append([c/255 for c in hex_to_rgb(color_dict[color_hex])])
    return LinearSegmentedColormap.from_list(name, colors, N)

def ow_colormap(color = 'ow_standard'):
    if color == 'ow_standard':
        return colormap_linear(['dark_blue', 'blue', 'gray', 'light_gray', 'dark_green', 'green'], name=color)
    elif color == 'ow_full':
        return colormap_linear(['blue', 'gray', 'green', 'teal', 'turquoise', 'yellow', 'orange', 
                                'crimson', 'pink', 'purple', 'blue_gray'], name=color)
    elif color == 'ow_all':
        return colormap_linear(list(color_dict.keys()), name=color)
    elif color == 'blues':
        return colormap_linear(['dark_blue', 'light_blue'], name=color)
    elif color == 'greens':
        return colormap_linear(['dark_green', 'light_green'], name=color)
    elif color in ['grays','greys']:
        return colormap_linear(['dark_gray', 'light_gray'], name='grays')
    elif color == 'teal':
        return colormap_linear(['dark_teal', 'light_teal'], name=color)