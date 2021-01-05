<img src='/doc/images/teaser_fig.png'>

### To Do
- This readme
  - Go through the English
  - Add links
- Set up dropbox for all datasets
- Scripts to download datasets
- Auto process datasets
- **Train all / each dataset**
- Evaluate all / each dataset
- Add pretrained models to dropbox
- Improve this README: add video & images, ..
- Self-code-review:
  - dataset_prepare
  - evaluate X 2

# MeshWalker: Deep Mesh Understanding by Random Walks
Created by [Alon Lahav](mailto:alon.lahav2@gmail.com).

## Installation
A step-by-step installation guide for Ubuntu is provided in [INSTALL.md](./INSTALL.md).

## Data
<img src='/doc/images/segmentaion.gif'>
Note for this README: each time `<dataset>` is mentioned, 
it should be replaced by one of the following:
```
1. modelnet40
2. engraved_cubes
3. shrec11
4. coseg
5. human_seg
```
You can also use `all` instead of a specific dataset.

### Raw datasets
To get the raw datasets go to the relevant website, 
and put it under `~/datasets/<dataset>`. 
- [ModelNet](https://modelnet.cs.princeton.edu/)
  (Right click on `ModelNet40.zip`, to download the dataset) 
- [Shrec11]():
- [Engraved Cubes]():
- [Human-seg17]():
- [COSEG]():

You can also download it from our [raw_datasets]() folder.
Please run `bash ./scripts/download_raw_datasets.sh`.


### Processed
To prepare the data, run `python dataset_prepare.py <dataset>`

Or download the data after processing. 
Processing will rearrang dataset in `npz` files, labels included, vertex niebours added.
```
bash ./scripts/download_processed_<dataset>.sh
```
 
## Training
```
python train_val.py <dataset>
```
You will find the results at: `~\mesh_walker\runs\<dataset>`

Use tensorboard to show training results: `tensorboard <trained-model-folder>`

Note that "accuracy" tab does not...

<img src='/doc/images/2nd_fig.png'>

## Evaluating
After training is finished, run the following to get accuracy results: `python evaluate.py \<dataset> <trained-model-folder>`

## Pretrained
You can use some [pretrained](https://technionmail-my.sharepoint.com/personal/alon_lahav_campus_technion_ac_il/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Falon%5Flahav%5Fcampus%5Ftechnion%5Fac%5Fil%2FDocuments%2Fmesh%5Fwalker%2Fpretrained)  models to run evaluation only. 

## Troubleshooting
If rendering using `opengl` doesn't work, 
it might be because `LANG` environment parameter is not set to `en_US`.

To fix it just write: `LANG=en_US` in command line and then run the python script.

If you use PyCharm, go to: `Run -> Edit Configurations...` and add `
;LANG=en_US` to the `Environment variables:`