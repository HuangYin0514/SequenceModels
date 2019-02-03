# -*- coding: utf-8 -*-
# @Time     : 2019/2/3 13:56
# @Author   : HuangYin
# @FileName : main.py
# @Software : PyCharm

from week3.q2.method import *

if __name__ == '__main__':

    print("Time steps in audio recording before spectrogram", data[:,0].shape)
    print("Time steps in input after spectrogram", x.shape)

    print("background len: " + str(len(backgrounds[0])))    # Should be 10,000, since it is a 10 sec clip
    print("activate[0] len: " + str(len(activates[0])))     # Maybe around 1000, since an "activate" audio clip is usually around 1 sec (but varies a lot)
    print("activate[1] len: " + str(len(activates[1])))     # Different "activate" clips can have different lengths

    overlap1 = is_overlapping((950, 1430), [(2000, 2550), (260, 949)])
    overlap2 = is_overlapping((2305, 2950), [(824, 1532), (1900, 2305), (3424, 3656)])
    print("Overlap 1 = ", overlap1)
    print("Overlap 2 = ", overlap2)

    np.random.seed(5)
    audio_clip, segment_time = insert_audio_clip(backgrounds[0], activates[0], [(3790, 4400)])
    audio_clip.export("insert_test.wav", format="wav")
    print("Segment Time: ", segment_time)
    IPython.display.Audio("insert_test.wav")

    arr1 = insert_ones(np.zeros((1, Ty)), 9700)
    plt.plot(insert_ones(arr1, 4251)[0,:])
    print("sanity checks:", arr1[0][1333], arr1[0][634], arr1[0][635])

    x, y = create_training_example(backgrounds[0], activates, negatives)

    model = model(input_shape=(Tx, n_freq))
    model.summary()

    opt = Adam(lr=0.0001,beta_1=0.9,beta_2=0.999,decay=0.001)
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.fit(X,Y,batch_size=5,epochs=1)

    loss,acc = model.evaluate(X_dev, Y_dev)
    print("Dev set accuracy = ", acc)

    filename = "raw_data/dev/1.wav"
    prediction = detect_triggerword(filename)
    chime_on_activate(filename, prediction, 0.5)
    IPython.display.Audio("chime_output.wav")

    filename  = "raw_data/dev/2.wav"
    prediction = detect_triggerword(filename)
    chime_on_activate(filename, prediction, 0.5)
    IPython.display.Audio("chime_output.wav")

    your_filename = "audio_examples/my_audio.wav"
    preprocess_audio(your_filename)
    IPython.display.Audio(your_filename)  # listen to the audio you uploaded
    chime_threshold = 0.5
    prediction = detect_triggerword(your_filename)
    chime_on_activate(your_filename, prediction, chime_threshold)
    IPython.display.Audio("./chime_output.wav")

