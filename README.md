### Github Repo Structure:
- DataTesting: Contains All Audio Sample analysis and Tests
- Transcription: Contains Scripts for various attempted transcriptions
- Logs: Contains transcription results and datatesting results
- Experiments: Contains various text grammar reasoning method implementation
- Inference: Contains scripts for Infer time run for various experiments
- Checkpoints: Contains Model Checkpoints on Training Data.

## Project Documentation: Grammar Scoring Engine

This project focused on building a grammar scoring engine for audio samples using a combination of speech-to-text transcription and fine-tuned language models.

 ### Data Exploration

The first step involved analyzing the provided audio files—examining sampling rates, durations, and waveforms. Most samples had durations clustered around 60 seconds, with some extending up to 180 seconds. However, samples exceeding 70 seconds often exhibited substantial noise—likely due to recording artifacts or sampling inconsistencies. These samples were found to contribute poorly to transcription quality and were excluded from training to maintain the integrity of downstream tasks.

 ### Audio Transcription Pipeline

The initial transcription pipeline tested  Wenet Gigaspeech with LM (n-gram based decoding). While it handled different accents reasonably well, it often missed key words or produced incoherent phrases. Given that transcription quality directly impacts grammar scoring, this method was deemed insufficient.

Subsequently, OpenAI's Whisper was evaluated and demonstrated significant improvement in transcription accuracy, especially in capturing natural speech. Since Whisper processes only short (30s) audio chunks, a custom chunking and post-processing pipeline was implemented. However, inference time proved to be a bottleneck.

To address this, Faster-Whisper was adopted, achieving near-equivalent transcription quality with dramatically faster inference and reduced memory usage. This became the backbone of the transcription component.

 ### Grammar Scoring via Language Models

For grammar evaluation, multiple strategies were tested:

FLAN-T5 and BERT models were fine-tuned on the dataset, using transcriptions as input and grammar rubric scores as targets.

MPNet was evaluated using a cosine similarity-based strategy between transcription embeddings and rubric descriptions. While conceptually interesting, it did not yield reliable performance and was ultimately dropped.

LanguageTool-Python was used to extract grammar error features and attempted as part of an ensemble approach. However, noisy transcriptions often led to misleading grammar assessments, making the integration counterproductive.

Fine-tuning BERT and T5 on rubric-annotated data led to the most promising results. The final evaluation on the held-out test set achieved a score of 0.827 (on 30% of the test set).

 ### Checkpoints (RMSE Metric)
Bert Finetuned on Training Data
      "epoch": 50.0,
      "grad_norm": 1.4541561603546143,
      "learning_rate": 1.5000000000000002e-08,
      "loss": 0.0133,
      "step": 2000
      "eval_loss": 0.4391864240169525,
      "eval_runtime": 0.273,
      "eval_samples_per_second": 131.878,
      "eval_steps_per_second": 18.316,
      "step": 2000

 ### Additional Handling

For audios exceeding 70 seconds in duration, the transcription is flagged and not processed further.

In special inference conditions (e.g., failed or flagged transcription), label assignment is handled probabilistically with a weighted strategy to reflect realistic scoring distributions.

##### There are several planned improvements for the future after ongoing endsems:

Better error-aware postprocessing for transcription.

Grammar-specific transformers or multi-task learning.

Self-supervised grammar error correction pipelines.

Denoising or filtering problematic audio samples to extend the usable dataset.
