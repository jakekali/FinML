import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

class data_reader:
    file_name = ""
    df = None
    buy = None
    hold = None
    sell = None
    images_buy = None
    images_hold = None
    images_sell = None

    def __init__(self, file_name):

        if file_name == "":
            raise ValueError("File name is empty")

        self.file_name = file_name
        self.df = pd.read_csv(file_name, header=None, sep=';')

        self.df = self.df.iloc[15:,:]

        self.df.columns = ["label", "price"] + ["pixel" + str(i) for i in range(0, 226)]
        self.df.drop(["price", "pixel225"], axis=1, inplace=True)
        self.df.dropna(inplace=True)

        self.buy = self.df[self.df['label'] == 1]
        self.hold = self.df[self.df['label'] == 0]
        self.sell = self.df[self.df['label'] == 2]

        self.buy = self.buy.drop(["label"], axis=1)
        self.hold = self.hold.drop(["label"], axis=1)
        self.sell = self.sell.drop(["label"], axis=1)

        self.images_buy = np.array(self.buy.values).reshape(-1, 15, 15)
        self.images_hold = np.array(self.hold.values).reshape(-1, 15, 15)
        self.images_sell = np.array(self.sell.values).reshape(-1, 15, 15)


    def showImages(self, count):
        
        # randomly select count images from each category
        images_buy = self.images_buy[np.random.choice(self.images_buy.shape[0], count, replace=False)]
        images_hold = self.images_hold[np.random.choice(self.images_hold.shape[0], count, replace=False)]
        images_sell = self.images_sell[np.random.choice(self.images_sell.shape[0], count, replace=False)]



        if count < 1:
            raise ValueError("Count must be greater than 0")
        
        plt.figure(figsize=(count*2,6.5))
        # Add a title to the rows
        plt.suptitle(self.file_name, fontsize=16)

        for i in range(0, count):
            plt.subplot(3, count, i + 1)
            plt.imshow(images_buy[i], cmap=plt.get_cmap('gray'))
            if i == 0:
                plt.title("Buy")

        for i in range(0, count):
            plt.subplot(3, count, i + 1 + count)
            plt.imshow(images_hold[i], cmap=plt.get_cmap('gray'))
            if i == 0:
                plt.title("Hold")

        for i in range(0, count):
            plt.subplot(3, count, i + 1 + count + count)
            plt.imshow(images_sell[i], cmap=plt.get_cmap('gray'))
            if i == 0:
                plt.title("Sell")

        # add timestap to the left bottom corner
        plt.figtext(0.05, 0.05, "Created by: data_reader.py", color="black", fontsize=10, ha="left", va="bottom", alpha=0.5)
        plt.figtext(0.95, 0.05, time.strftime("%d.%m.%Y %H:%M:%S"), color="black", fontsize=10, ha="right", va="bottom", alpha=0.5)

        plt.savefig("images/" + self.file_name + ".png")

    def createTFDataset(self, upSample=True):
        import tensorflow as tf

        buy = self.images_buy
        hold = self.images_hold
        sell = self.images_sell

        if upSample:
            class_counts = np.array([self.buy.shape[0], self.hold.shape[0], self.sell.shape[0]])
            print(class_counts)
            ratio = np.max(class_counts)//class_counts
            print(ratio)

            buy = np.repeat(self.images_buy, ratio[0], axis=0)
            hold = np.repeat(self.images_hold, ratio[1], axis=0)
            sell = np.repeat(self.images_sell, ratio[2], axis=0)

            print("buy shape: ", buy.shape)
            print("hold shape: ", hold.shape)
            print("sell shape: ", sell.shape)


        data = np.concatenate((buy, hold, sell), axis=0)
        labels = np.concatenate((2 * np.ones(buy.shape[0]), 1* np.ones(hold.shape[0]), 0* np.ones(sell.shape[0])), axis=0)

        dataset =  tf.data.Dataset.from_tensor_slices((data, labels))
        dataset = dataset.shuffle(buffer_size=9000).batch(1024)

    
        return dataset
    

    def stats_generator(self):

        print("File name: ", self.file_name)
        print("Number of samples: ", self.df.shape[0])

        num_hold = self.hold.shape[0]
        num_buy = self.buy.shape[0]
        num_sell = self.sell.shape[0]

        print("Number of hold: ", num_hold)
        print("Number of buy: ", num_buy)
        print("Number of sell: ", num_sell)

        print("Hold percentage: ", num_hold/(num_hold + num_buy + num_sell))
        print("Buy percentage: ", num_buy/(num_hold + num_buy + num_sell))
        print("Sell percentage: ", num_sell/(num_hold + num_buy + num_sell))

        print("Total number of samples: ", (num_hold + num_buy + num_sell))

    



if __name__ == "__main__":
    data_reader("outputOfPhase2Training.csv").showImages(9)