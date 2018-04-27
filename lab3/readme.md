# Usage

Part 1 - Show, Attend and tell:
```
python3 train.py --caption_model show_attend_tell --batch_size 4 --epoch 10
```

Part 2 - Top down:

```
python3 train.py --caption_model top_down --batch_size 4 --epoch 10
```

Bonus - Where to look:

```
python3 train.py --caption_model where_to_look --batch_size 4 --epoch 10
```

Evaluate performance:
* You should ensure the location of `val2014` folder is in `./`
```
python3 eval.py --model show_attend_tell_model.pth --infos_path show_attend_tell_infos.pkl
```

Visualize caption:
```
python3 visualize.py --image_folder visualize_img --caption_model show_attend_tell
```