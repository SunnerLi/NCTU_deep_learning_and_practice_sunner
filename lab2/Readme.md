Usage
---

Requirement 1:
```
$ python3 4_plot.py --image_path ./images/SR_GT.png --epoch 2400
```

Requirement 2:
```
$ python3 denoising.py --noise_img_gt images/noise_GT.png --epoch 1800
```

Requirement 3:
```
$ python3 sr.py --gt ./images/SR_GT.png --epoch 2000
```

Bonus:
```
$ python3 bonus.py --epoch 3001
```