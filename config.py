# Pré-processamento de imagem
BLUR_KERNEL = 5
THRESHOLD = 100

# Canny
CANNY_LOW = 50
CANNY_HIGH = 150

# Hough
HOUGH_THRESHOLD = 100
MIN_LINE_LENGTH = 50
MAX_LINE_GAP = 20

# Wall detection
# tolerância para considerar linhas colineares (px)
COLINEAR_TOLERANCE = 30

# Gap máximo para unir segmentos da mesma parede
MERGE_GAP = 120

# Comprimento mínimo final de uma parede
MIN_WALL_AXIS_LENGTH = 80

# pixels – ajuste fino depois
MAX_DIMENSION_OFFSET = 40
