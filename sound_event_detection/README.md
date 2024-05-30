First extract data:
```bash
cd data;
sbatch prepare_data.sh /dssg/home/acct-stu/stu464/data/domestic_sound_events
cd ..;
```

Then run with best setting:
```bash
python run.py train_evaluate configs/best.yaml 
```