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

# distância máxima entre linhas paralelas para colapsar (px)
MAX_WALL_THICKNESS = 40

# porcentagem mínima de sobreposição (0–1)
MIN_OVERLAP_RATIO = 0.6
