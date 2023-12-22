import turtle

def draw_triangle(points, color, my_turtle):
    my_turtle.fillcolor(color)
    my_turtle.up()
    my_turtle.goto(points[0][0], points[0][1])
    my_turtle.down()
    my_turtle.begin_fill()
    my_turtle.goto(points[1][0], points[1][1])
    my_turtle.goto(points[2][0], points[2][1])
    my_turtle.goto(points[0][0], points[0][1])
    my_turtle.end_fill()

def midpoint(point1, point2):
    return ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)

def sierpinski_triangle(points, degree, my_turtle):
    color_map = ['blue', 'red', 'green']
    draw_triangle(points, color_map[degree], my_turtle)
    if degree > 0:
        sierpinski_triangle([points[0],
                             midpoint(points[0], points[1]),
                             midpoint(points[0], points[2])],
                            degree-1, my_turtle)
        sierpinski_triangle([points[1],
                             midpoint(points[0], points[1]),
                             midpoint(points[1], points[2])],
                            degree-1, my_turtle)
        sierpinski_triangle([points[2],
                             midpoint(points[2], points[1]),
                             midpoint(points[0], points[2])],
                            degree-1, my_turtle)

def main():
    my_turtle = turtle.Turtle()
    my_win = turtle.Screen()
    my_points = [[-200, -100], [0, 200], [200, -100]]
    sierpinski_triangle(my_points, 3, my_turtle)
    my_win.exitonclick()

if __name__ == "__main__":
    main()
