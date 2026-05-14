# Preemtpible job tutorial

## Step 1: Login to Sherlock Shell
Either via [OnDemand](https://ondemand.sherlock.stanford.edu/pun/sys/dashboard)

Or via ssh: `ssh <sunet>@login.sherlock.stanford.edu`

## Step 2: Clone this Repository

```bash
git clone https://github.com/MATRICS-Bootcamp/preempting-and-checkpointing.git
cd preempting-and-checkpointing;
```

## Step 3: Copy the container over

```bash
apptainer pull oras://ghcr.io/matrics-bootcamp/pytorch:latest
```

If this takes too long, you can copy from scratch:
```bash
cp /oak/stanford/schools/ees/containers/pytorch_latest.sif ./env/
```

## Step 4: Submit the Job

```bash
cd job;
sbatch preemptible_job.submit
```

## Step 5: Watch the Job

```bash
watch -n 20 squeue --me
```

And maybe chat a bit about checkpointing!

## Step 6: PreEmpt the Job

```bash
scontrol requeue <job_id>
```

## Step 7: Observe again!

Watch the squeue as your job requeues.  Also - watch the output and models directory to confirm it restarted from the checkpoint.
