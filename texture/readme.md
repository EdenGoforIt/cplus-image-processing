# Texture Classification

Simple texture classification using LBP and kNN to segment grass, clouds, and sea.

## Setup

```
git clone <repository-url>
cd texture
mkdir build && cd build
cmake ..
make
```

## Data Folder

Put your data in these folders:

- `data/grass/` - Grass images for training
- `data/cloud/` - Cloud images for training
- `data/sea/` - Sea images for training
- `data/case1.jpg` - Case 1 Image with Grass + Clouds
- `data/case2.jpg` - Case 2 Image with Grass + Sea
- `data/case3.jpg` - Case 3 Image with Grass + Sea + Clouds
- `data/case4.npg` - Case 4 Image with Grass + Sea + Clouds
- `data/result.png` - The result of class segmentation for case 1,2,3,4

## Run

```
./src/main case1.jpg
```

Colors in output:

- Green: Grass
- Grey: Clouds
- Blue: Sea
