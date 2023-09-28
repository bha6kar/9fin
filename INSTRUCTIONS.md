# 9fin Senior ML Software Engineer Technical Challenge

Welcome to the challenge!

When you are ready to go through what you've done, email me back, and we can schedule the technical interview. There is no time limit here but please don't spend more than 1.5-2 hours on this task. If you feel stuck for some time or there is an issue with the challenge, please stop and send me a quick email to help you out.

The aim of this challenge is to provide something for us to talk around in the interview. We want to test your ability to write clean performant inference code for this ML model artifact using best SE practices.

We can always discuss any further plans if there was more time to iterate on this.

Good luck!

## Problem

Our Product Managers have asked the DS/ML team to see how feasible it is to introduce a form of semantic search into our platform. We've relied heavily on sparse keyword search over the years but we've noticed a large number of customers don't know what they are looking for. 
One of our applied scientist colleagues has been deep into data and literature trying to train a model to improve upon the open source baseline using our private evaluation set. The PMs want to evaluate this new approach against our existing search product live on production and that requires a few things e.g getting model to production, setting up offline ingestion and online querying mechanisms.

## Task

Your colleague has passed you a `predict.py` with their model prediction function code and they have also added a gzip compressed archive file containing the model directory.

Let's look at building the offline ingestion function which takes a document identifier (str) and returns a list of vectors with their relevant metadata. We will encode our document texts using the provided sentence transformer model and enrich the vectors with the companying page numbers. 

The function should would return a list resembling the following:
```
{
  'text': str,
  'vector': torch.Tensor,
  'pages': list[int]
}
```

The document identifier will allow us to identify and use the `pages_[doc_id].csv` (unique page references to actual page numbers) and `texts_[doc_id].json` (text and page ref array).

For the case of this take home, we can assume that all the text/pages/model files are located somewhere on the local filesystem (within the container of course). 

We will want to execute this, via Docker, using the following script i.e.
```python
if __name__ == '__main__':
  doc_id = '12345'
  your_func(doc_id)
```

Commit your files to a private git repo and share it with us (or upload your code repo with git dir to the same shared drive) when you're ready. Up to you here!  My github is https://github.com/SeanBE :)


## Questions

We've listed some of the questions that you can think about for the interview.

- How did you approach this task? What were all the things you could have done?
- If we were to deploy this to serverless or persistent instance on the cloud, what would we have to consider?
- What would you tell your colleague for the future (in terms of improved collaboration/reduced repeated work)
- If we added performance constraints on the docker container (or serverless function), what would happen to your application / what would we need to consider changing (e.g 1.5 CPU / 2GB MEM)?
- If we decided to use a managed store for all these vectors, how we go about saving these enriched vectors efficiently? 
