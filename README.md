# Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro Cog Model

This is an implementation of [Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro](https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro) as a [Cog](https://github.com/replicate/cog) model.

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own model to [Replicate](https://replicate.com).


## How to use

Make sure you have [cog](https://github.com/replicate/cog) installed.

To run a prediction:

    cog predict -i prompt="A bohemian-style female travel blogger with sun-kissed skin and messy beach waves" -i control_type="pose" -i control_image=@openpose.jpg

![Output](output.0.png)