import random

positive_prompt = ['a bright and clean photo.',
                   'The image is sharp and pristine, with every detail crisp and distinct.',
                   'Colors are vibrant and true, capturing the scene with vivid clarity.',
                   "There's a perfect clarity in the image, free from any haze or distortion.",
                   'Light and shadows play naturally, enhancing the scene without any blur.',
                   "Objects appear vivid and lifelike against a clear backdrop.",
                   "The scene is captured with absolute clarity, showcasing fine textures and details.",
                   "Each element stands out sharply, from foreground to distant horizon.",
                   "The air quality in the image is excellent, allowing for a clear view of distant landmarks.",
                   "It's a picture-perfect moment, with no haziness to obscure the view.",
                   "Every detail is sharp and well-defined, offering a clear and immersive visual experience."
                   "a photo with high resolution and sharp details.",
                   "a photo without haze or fog.","Textures and structures are clear and not obscured by fog.",
                   "The image is sharp and detailed, with no haze or fog to obscure the scene.",
                   "The photo is crisp and clear, with every detail visible and distinct.",
                   "The images are clear and the colors are vivid.",
                   "The image quality is very good, and nothing is obscured by fog.",
                   "The landscape and objects in the image are crystal clear without any fog.",
                   "The pedestrians and objects in the image are clear and not obscured by fog.",
                    "The image is crisp and sharp, with every detail rendered in high definition.",
                   "The scene is free of haze, allowing the viewer to see every nuance and texture.",
                   "The image is bathed in vibrant colors, without a hint of murkiness or fog.",
                   "The details are precise and accurate, with no room for misinterpretation.",
                   "The image has a sense of clarity, making it easy to discern every element.",
                   "The fog-free atmosphere allows the viewer to see the scene in all its glory.",
                   "The image is free of noise, with every pixel precise and detailed.",
                   "The colors are rich and saturated, without a hint of washed-out or muted tones.",
                   "The image has a sense of depth and dimensionality, drawing the viewer into the scene.",
                   "The scene is rendered in exquisite detail, with every element perfectly placed.",
                   "The image is razor-sharp, with every detail in perfect focus.",
                   "The colors are rich and intense, with a sense of depth and dimension.",
                   "The lighting is perfect, with subtle shading and nuanced texture.",
                   "The photo is free from any obstacles, offering an unimpeded view.",
                   "Every aspect of the image is meticulously rendered, with no detail overlooked.",
                   "he photo glows with a natural light, making it seem almost lifelike.",
                   "The photo is completely free of imperfections, with a sense of freshness and vitality.",
                   "The photo is so clear, it's almost mesmerizing.",
                   "Every aspect of the image is rendered in exquisite detail, with no loss of resolution.",
                   "The photo remains crystal-clear, even when viewed at close range.",
                   "The colors are so vibrant, they seem to pulse with energy.",
                   "Every element of the image is precisely rendered, with no room for error.",
                   "The image is relentlessly clear, with no concessions to ambiguity.",
                   "The image is a testament to the power of precision, with every detail meticulously rendered.",
                   "The image offers a crystal-clear view of the world, unfiltered and unadorned.",
                    "The photo is a masterpiece of clarity, with every detail perfectly preserved.",
                    "The image is a study in precision, with every element perfectly placed.",
                    "The photo is a testament to the power of detail, with every pixel perfectly rendered.",
                    "The image is a symphony of clarity, with every note perfectly struck.",
                    "The photo is a masterpiece of precision, with every detail meticulously rendered.",
                    "The image is a testament to the power of clarity, with every element perfectly preserved.",
                   ]

negative_prompt = ["a dirty and dark photo.", "a photo with noisy marks and artifacts.", "a photo with low contrast.",
                   "a photo with JPEG compression artifacts.",
                   "The image is dark and murky, with details obscured by shadows and noise.",
                   "the picture looks gray and has low contrast.",
                   "The image is filled with noise and artifacts, obscuring the scene with digital distortion.",
                   "The photo has a low contrast, making it difficult to distinguish details.",
                   "The image is heavily compressed, with visible artifacts and distortion.",
                   "The photo is dark and underexposed, with details lost in the shadows.",
                   "The image is marred by digital noise, obscuring the scene with visual interference.",
                   "The photo has a low resolution, with visible pixelation and blurring.",
                   "The image is heavily compressed, with visible artifacts and distortion.",
                   "The photo is dark and underexposed, with details lost in the shadows.",
                   "The image is marred by digital noise, obscuring the scene with visual interference.",
                   "a photo with haze and fog.","The structure of the image is blurred and the texture is unclear.",
                   "The image is obscured by haze and fog, with distant objects barely visible.",
                   "The photo is hazy and indistinct, with details obscured by a thick fog.",
                   "The image is obscured by haze and fog, with distant objects barely visible.",
                    "The photo is hazy and indistinct, with details obscured by a thick fog.",
                   "The image appears grayish and the colors are not vibrant.",
                    "The image is blurry and the details are not clear.",
                   "The image quality is poor, with content and structures obscured by fog.",
                    "The image is blurry and the details are not clear.",
                   "The image with haze, the details are not clear.",
                   "the photo have fog and haze, the details are not clear.",
                   "The image is shrouded in a thick layer of fog, obscuring all details.",
                   "A misty veil hangs over the scene, casting a mysterious gloom.",
                   "The fog is so dense that it's hard to make out any shapes or forms.",
                   "All textures and patterns are blurred and indistinct, giving the image a featureless quality.",
                     "The image is shrouded in a thick layer of fog, obscuring all details.",
                     "A misty veil hangs over the scene, casting a mysterious gloom.",
                   "The image is heavily affected by noise, with random pixels scattered throughout.",
                   "The fog is so dense that it's hard to make out any details at all.",
                   "The image is suffering from a serious case of blur, making it hard to distinguish between shapes.",
                   "The image is so vague and indistinct that it's hard to tell what's going on in the scene.",
                   "The fog has obscured all the important details, leaving the image feeling empty and useless to the viewer.",
                   "The image is plagued by chromatic aberration, making colors look fuzzy and unclear.",
                   "The image is suffering from a lack of contrast, making it hard to distinguish between different elements.",
                   "The fog has created a sense of disorientation, making it hard to tell which direction is up or down.",
                   "The image is so poorly lit that it's hard to see anything at all, let alone make out any details.",
                     "The image is so poorly composed that it's hard to tell what the subject is supposed to be.",
                   "The edges of the image are obscured by fog and are not clear.",
                   "The image is suffering from a lack of texture, making it feel flat and uninteresting.",
                    "The image is so poorly exposed that it's hard to see any details at all.",
                   "The image is plagued by artifacts, making it look like a poor quality print.",
                   "The fog has created a sense of disorientation, making it hard to tell what's real and what's not.",
                   "The image is so poorly composed that it's hard to follow the action.",
                   "The fog has created a sense of confusion.",
                   ]


enhance_prompts = []

p1 =  random.choice(positive_prompt)

p2 = selected_from_B = random.sample(negative_prompt, min(5, len(negative_prompt)))  #返回一个数组

enhance_prompts.append(p1)

enhance_prompts.extend(p2)

print(enhance_prompts)

def RANSAC(sample):

    selected_from_A = random.choice(positive_prompt) #返回一个字符串

    selected_from_B = random.sample(negative_prompt, min(1, len(negative_prompt)))  #返回一个数组

    while(True):
        if selected_from_B not in sample:
            break
        else:
            selected_from_B = random.sample(negative_prompt, min(1, len(negative_prompt)))

    neg = sample[1:]

    negative = selected_from_B + neg

    return [selected_from_A] + negative


def RANSAC():


    selected_element = random.choice(positive_prompt)  # 从 list1 中选取一个元素
    selected_elements = random.sample(negative_prompt, 10)  # 从 list2 中随机选取 10 个元素
    new_list = [selected_element] + selected_elements  # 组合成新列表

    return new_list







