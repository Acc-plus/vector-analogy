import svgwrite
from svgwrite.shapes import Line, Circle, Polygon, Rect
from svgwrite.path import Path
import numpy as np
import numpy.random as random
from svgpathtools import parse_path

stroke_width = 25

class primitive():
    def __init__(self):
        pass
    
    def svgcode(self):
        pass

    def params(self):
        pass

    



class rectangle(primitive):
    def __init__(self, dx, dy, width, height, fill = [30, 40, 50]):
        self.dx = dx
        self.dy = dy
        self.width = width
        self.height = height
        self.concat_spot = []
        self.spot_order = []
        self.spots = 0
        self.add_isometric_concat_spot(3)
        self.fill = fill
    
    def clone(self):
        return rectangle(self.dx, self.dy, self.width, self.height, self.fill)

    def change_color(self, fill):
        self.fill = fill

    def add_concat_spot(self, x, y):
        self.concat_spot.append([x, y])
        
    def add_isometric_concat_spot(self, frequency):
        if (self.width > self.height):
            interval = self.width / (frequency - 1)
            for metric in range(frequency):
                self.concat_spot.append([metric * interval + self.dx, self.dy])
        else:
            interval = self.height / (frequency - 1)
            for metric in range(frequency):
                self.concat_spot.append([self.dx, metric * interval + self.dy])
        self.spot_order = [i for i in range(frequency)]
        random.shuffle(self.spot_order)
        # print(self.concat_spot)

    def choose_concat(self):
        lcs = len(self.concat_spot)
        choice = np.random.randint(0, lcs)
        return self.concat_spot[choice]

    def svgcode(self):
        return svgwrite.shapes.Rect((self.dx, self.dy), (self.width, self.height), 
            fill=svgwrite.rgb(*self.fill, '%'))

    def params(self):
        return np.array([self.dx, self.dy, self.width, self.height, self.fill[0], self.fill[1], self.fill[2]])

    def __repr__(self):
        return f'{self.dx}, {self.dy}, {self.width}, {self.height}, fill = {self.fill}'

def texurize_rectangle_add_tail_squre(rec : rectangle):
    if (rec.width > rec.height):
        return rectangle(rec.dx + rec.width - stroke_width, rec.dy, stroke_width, stroke_width, 
            fill=[60, 40, 20])
    else:
        return rectangle(rec.dx, rec.dy + rec.height - stroke_width, stroke_width, stroke_width, 
            fill=[60, 40, 20])


class triangle(primitive):
    def __init__(self):
        pass

class SVG():
    def __init__(self):
        self.primitives = []
        self.successors = []

    def svgparams(self):
        params = []
        for prim in self.primitives:
            params.append(prim.params())
        return np.array(params)

    def clone(self):
        svg = SVG()
        for prim in self.primitives:
            svg.primitives.append(prim.clone())
        for succ in self.successors:
            svg.successors.append(succ)
        return svg
    


    
def construct_random_svg():
    strokes = random.randint(3, 6);
    random.rand()
    svg = SVG()
    pdx = random.randint(50, 100)
    pdy = random.randint(50, 100)
    pw = random.randint(100, 300)
    ph = random.randint(100, 300)
    if (pw < ph):
        pw = stroke_width
    else:
        ph = stroke_width
    primary_rec = rectangle(pdx, pdy, pw, ph)
    svg.primitives.append(primary_rec)
    svg.successors.append(0)
    primative_ptr = 0
    successor_ptr = 0
    while strokes > 0:
        strokes -= 1
        temp_prim : rectangle = svg.primitives[primative_ptr]
        # print(temp_prim)
        suc_ptr = temp_prim.spot_order[svg.successors[successor_ptr]]
        spot = temp_prim.concat_spot[suc_ptr]
        if (temp_prim.width < temp_prim.height):
            w = random.randint(100, 300)
            h = stroke_width
        else:
            w = stroke_width
            h = random.randint(100, 300)
        dx = spot[0]
        dy = spot[1]
        svg.primitives.append(rectangle(dx, dy, w, h))
        svg.successors.append(0)
        svg.successors[successor_ptr] += 1
        if (random.rand() > 0.5) or \
            svg.successors[successor_ptr] >= len(temp_prim.spot_order):
            primative_ptr += 1
            successor_ptr += 1
    return svg

def construct_random_svg_horizontal_rectangle():
    strokes = random.randint(3, 6);
    random.rand()
    svg = SVG()
    pdx = random.randint(50, 100)
    pdy = random.randint(50, 100)
    pw = random.randint(100, 300)
    ph = stroke_width
    primary_rec = rectangle(pdx, pdy, pw, ph)
    svg.primitives.append(primary_rec)
    svg.successors.append(0)
    primative_ptr = 0
    successor_ptr = 0
    for i in range(strokes):
        temp_prim : rectangle = svg.primitives[primative_ptr]
        # print(temp_prim)
        w = random.randint(100, 300)
        h = stroke_width
        svg.primitives.append(rectangle(random.randint(50, 150), 
            random.randint(i*100,(i+1)*100), w, h))
        svg.successors.append(0)
        svg.successors[successor_ptr] += 1
        if (random.rand() > 0.5) or \
            svg.successors[successor_ptr] >= len(temp_prim.spot_order):
            primative_ptr += 1
            successor_ptr += 1
    return svg

def draw_rec(rec : rectangle, file_svg = 'rec.svg'):
    dwg = svgwrite.Drawing(file_svg, profile='full')
    dwg.add(rec.svgcode())
    dwg.viewbox(0, 0, 500, 500)
    dwg.save()

def draw_rec_from_net_params(params, file_svg = 'rec.svg'):
    print(params)
    rec = rectangle(params[0], params[1], params[2], params[3], 
        [params[4], params[5], params[6]])
    draw_rec(rec, file_svg)

def draw_svg(svg : SVG, file_svg = 'test.svg'):
    dwg = svgwrite.Drawing(file_svg, profile='full')
    for prims in svg.primitives:
        dwg.add(prims.svgcode())
    dwg.viewbox(0, 0, 500, 500)
    dwg.save()

def texturize_svg(svg : SVG, file_svg = 'test.svg'):
    texture = []
    for prims in svg.primitives:
        texture.append(prims)
        texture.append(texurize_rectangle_add_tail_squre(prims))
    texvg = SVG()
    texvg.primitives = texture
    return texvg

def TrainData_SVG(n_svg = 100):
    svgs = [construct_random_svg_horizontal_rectangle() for _ in range(n_svg)]
    # texvgs = [texturize_svg(svg) for svg in svgs]
    texvgs = [svg.clone() for svg in svgs]
    for svg in texvgs:
        for prims in svg.primitives:
            prims.fill = [20, 60, 100]

    # import pdb
    # pdb.set_trace()
    return svgs, texvgs



def TestData_SVG():
    pass

if __name__ == '__main__':
    # print(parse_path('M0 0 H 100 V 100 Q 50 50 0 100 Z'))
    # exit()
    # svg = construct_random_svg_horizontal_rectangle()
    # texvg = texturize_svg(svg)
    # draw_svg(svg)
    draw_rec_from_net_params([0,0,100,100,0,0,0])
