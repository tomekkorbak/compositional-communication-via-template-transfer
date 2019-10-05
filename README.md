Compositional communication via template transfer
==================================

This repo contains the code accompanying the paper *Developmentally motivated emergence of compositional communication via template transfer* accepted for NeurIPS 2019 workshop [Emergent Communication: Towards Natural Language](https://sites.google.com/view/emecom2019/home?authuser=0).

## Running the code

We assume Python <= 3.6. To reproduce the results of template transfer run:
```bash
pip install -r requirements.txt
unzip data.zip
python -m template_transfer.train
```

To reproduce reported baselines, run the following commands:
* Random baseline: `python -m template_transfer.train --sender_lr 0 --receiver_lr 0 --no_transfer`
* Same architecture without template transfer: `python -m template_transfer.train --no_transfer`
* Obverter: `python -m obverter.train`

Use `--help` flag for available arguments. All arguments default to hyperparameters used in the paper. I use [Neptune.ml](https://neptune.ml/) for experiment management, which is turned off by default. Pass `--neptune_project <username/projectname>` and set environmental variable `NEPTUNE_API_TOKEN` to log metrics using Neptune.

In case of questions, create an issue or contact Tomek Korbak under tomasz.korbak@gmail.com.

## Citing
```latex
@article{korbak_template_transfer_2019,
  author    = {Korbak, Tomasz and
               Zubek, Julian and
               Kuci\'{n}ski, \L{}ukasz and
               Mi\l{}oś, Piotr and
               R\k{a}czaszek-Leonardi, Joanna},
  title     = {Developmentally motivated emergence of compositional communication via template transfer},
  journal   = {NeurIPS 2019 workshop Emergent Communication: Towards Natural Language},
  year      = {2019},
  url       = {}
}
```
