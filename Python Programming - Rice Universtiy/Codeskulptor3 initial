# CodeSkulptor runs Python 3 programs in your browser.
# Click the upper left button to run this simple demo.

# CodeSkulptor runs in recent versions of Chrome, Firefox,
# and Safari.  Some features may work in other browsers,
# but do not expect full functionality.
# It does NOT run in Microsoft Internet Explorer or Edge.

import simplegui

message = "Welcome!"

# Handler for mouse click
def click():
    global message
    message = "Good job!"

# Handler to draw on canvas
def draw(canvas):
    canvas.draw_text(message, [50,112], 48, "Red")

# Create a frame and assign callbacks to event handlers
frame = simplegui.create_frame("Home", 300, 200)
frame.add_button("Click me", click)
frame.set_draw_handler(draw)

# Start the frame animation
frame.start()
