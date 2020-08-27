from source.sudoku import find_puzzle, extract_digits
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sudoku import Sudoku
import numpy as np
import imutils
import cv2

MODEL_PATH = "output/digit_classifier.h5"
IMAGE_PATH = "sudoku_test1.jpg"
# using debug for visualization of each step for better DEBUGGING obviously
debug = -1

# load the pre-trained model
print("[INFO] loading the model...")
model = load_model(MODEL_PATH)

# load and adjust the image
image = cv2.imread(IMAGE_PATH)
image = imutils.resize(image, width=600)

# find the puzzle in the image
(puzzleImage, warped) = find_puzzle(image, debug=debug > 0)

# initialize the 9x9 board
board = np.zeros((9, 9), dtype="int")

# infer the location of each cell by
# dividing the warped image into a 9x9 grid
stepX = warped.shape[1] // 9
stepY = warped.shape[0] // 9

# initialize a list to store the x, y coordinates of each cell
cellLocs = []

# loop over the grid locations
for y in range(0, 9):
    # initialize the current list of cell locations
    row = [] 

    for x in range(0, 9):
        # compute the starting and ending x, y coordinates of the current cell
        startX = x * stepX
        startY = y * stepY
        endX = (x + 1) * stepX
        endY = (y + 1) * stepY

        # add the x, y coordinates to the cell list 
        row.append((startX, startY, endX, endY))

        # crop the cell from the warped transform image and then extract the digits
        cell = warped[startY:endY, startX:endX]
        digit = extract_digits(cell, debug=debug > 0)

        # verify that the digit is not empty
        if digit is not None:
            foo = np.hstack([cell, digit])
            # cv2.imshow("Cell/Digit", foo)

            # resize the cell to 28x28 pixels and prepare it for classification
            roi = cv2.resize(digit, (28, 28))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # classify the digit and update the sudoku board
            pred = model.predict(roi).argmax(axis=1)[0]
            board[y, x] = pred
    
    # add the row to cell locations
    cellLocs.append(row)

# construct a sudoku puzzle 
print("[INFO] OCR'd sudoku board:")
puzzle = Sudoku(3, 3, board=board.tolist())
puzzle.show()

# solve the sudoku puzzle
print("[INFO] solving the sudoku puzzle")
solution = puzzle.solve()
solution.show_full()

# loop over cell locations and board
for (cellRow, boardRow) in zip(cellLocs, solution.board):
    # loop over individual cell in the row
    for (box, digit) in zip(cellRow, boardRow):
        # unpack the cell coordinates
        startX, startY, endX, endY = box

        # compute the coordinates of where the digit will be drawn on the image
        textX = int((endX - startX) * 0.33)
        textY = int((endY - startY) * -0.2)
        textX += startX
        textY += endY

        # draw the result digit on the sudoku image
        cv2.putText(puzzleImage, str(digit), (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

# show the output image
cv2.imshow("Sudoku Result", puzzleImage)
cv2.waitKey(0)
