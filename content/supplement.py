import matplotlib
import os
import subprocess
import urllib.request
import matplotlib.pyplot as plt
import gzip
import tarfile
import pickle
import numpy as np
try:
    import Image
except ImportError:
    from PIL import Image

from dezero_emb import *

class S_utils:
    # =============================================================================
    # Visualize for computational graph
    # =============================================================================
    def _dot_var(v, verbose=False):
        dot_var = '{} [label="{}", color=orange, style=filled]\n'

        name = '' if v.name is None else v.name
        if verbose and v.data is not None:
            if v.name is not None:
                name += ': '
            name += str(v.shape) + ' ' + str(v.dtype)

        return dot_var.format(id(v), name)


    def _dot_func(f):
        # for function
        dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
        ret = dot_func.format(id(f), f.__class__.__name__)

        # for edge
        dot_edge = '{} -> {}\n'
        for x in f.inputs:
            ret += dot_edge.format(id(x), id(f))
        for y in f.outputs:  # y is weakref
            ret += dot_edge.format(id(f), id(y()))
        return ret


    def get_dot_graph(output, verbose=True):
        txt = ''
        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                # funcs.sort(key=lambda x: x.generation)
                seen_set.add(f)

        add_func(output.creator)
        txt += _dot_var(output, verbose)

        while funcs:
            func = funcs.pop()
            txt += _dot_func(func)
            for x in func.inputs:
                txt += _dot_var(x, verbose)

                if x.creator is not None:
                    add_func(x.creator)

        return 'digraph g {\n' + txt + '}'


    def plot_dot_graph(output, verbose=True, to_file='graph.png'):
        dot_graph = get_dot_graph(output, verbose)

        tmp_dir = os.path.join(os.path.expanduser('~'), '.dezero')
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

        with open(graph_path, 'w') as f:
            f.write(dot_graph)

        extension = os.path.splitext(to_file)[1][1:]  # Extension(e.g. png, pdf)
        cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
        subprocess.run(cmd, shell=True)

        # Return the image as a Jupyter Image object, to be displayed in-line.
        try:
            from IPython import display
            return display.Image(filename=to_file)
        except:
            pass


    # =============================================================================
    # download function
    # =============================================================================
    def show_progress(block_num, block_size, total_size):
        bar_template = "\r[{}] {:.2f}%"

        downloaded = block_num * block_size
        p = downloaded / total_size * 100
        i = int(downloaded / total_size * 30)
        if p >= 100.0: p = 100.0
        if i >= 30: i = 30
        bar = "#" * i + "." * (30 - i)
        print(bar_template.format(bar, p), end='')

    
    def get_file(url, file_name=None):
        cache_dir = os.path.join(os.path.expanduser('~'), '.dezero_cache')
        if file_name is None:
            file_name = url[url.rfind('/') + 1:]
        file_path = os.path.join(cache_dir, file_name)

        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)

        if os.path.exists(file_path):
            return file_path

        print("Downloading: " + file_name)
        try:
            urllib.request.urlretrieve(url, file_path, S_utils.show_progress)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise
        print(" Done")

        return file_path


class transforms:

    class Compose:
        """Compose several transforms.

        Args:
            transforms (list): list of transforms
        """
        def __init__(self, transforms=[]):
            self.transforms = transforms

        def __call__(self, img):
            if not self.transforms:
                return img
            for t in self.transforms:
                img = t(img)
            return img


    # =============================================================================
    # Transforms for PIL Image
    # =============================================================================
    class Convert:
        def __init__(self, mode='RGB'):
            self.mode = mode

        def __call__(self, img):
            if self.mode == 'BGR':
                img = img.convert('RGB')
                r, g, b = img.split()
                img = Image.merge('RGB', (b, g, r))
                return img
            else:
                return img.convert(self.mode)


    class Resize:
        """Resize the input PIL image to the given size.

        Args:
            size (int or (int, int)): Desired output size
            mode (int): Desired interpolation.
        """
        def __init__(self, size, mode=Image.BILINEAR):
            self.size = utils.pair(size)
            self.mode = mode

        def __call__(self, img):
            return img.resize(self.size, self.mode)


    class CenterCrop:
        """Resize the input PIL image to the given size.

        Args:
            size (int or (int, int)): Desired output size.
            mode (int): Desired interpolation.
        """
        def __init__(self, size):
            self.size = utils.pair(size)

        def __call__(self, img):
            W, H = img.size
            OW, OH = self.size
            left = (W - OW) // 2
            right = W - ((W - OW) // 2 + (W - OW) % 2)
            up = (H - OH) // 2
            bottom = H - ((H - OH) // 2 + (H - OH) % 2)
            return img.crop((left, up, right, bottom))


    class ToArray:
        """Convert PIL Image to NumPy array."""
        def __init__(self, dtype=np.float32):
            self.dtype = dtype

        def __call__(self, img):
            if isinstance(img, np.ndarray):
                return img
            if isinstance(img, Image.Image):
                img = np.asarray(img)
                img = img.transpose(2, 0, 1)
                img = img.astype(self.dtype)
                return img
            else:
                raise TypeError


    class ToPIL:
        """Convert NumPy array to PIL Image."""
        def __call__(self, array):
            data = array.transpose(1, 2, 0)
            return Image.fromarray(data)


    class RandomHorizontalFlip:
        pass


    # =============================================================================
    # Transforms for NumPy ndarray
    # =============================================================================
    class Normalize:
        """Normalize a NumPy array with mean and standard deviation.

        Args:
            mean (float or sequence): mean for all values or sequence of means for
            each channel.
            std (float or sequence):
        """
        def __init__(self, mean=0, std=1):
            self.mean = mean
            self.std = std

        def __call__(self, array):
            mean, std = self.mean, self.std

            if not np.isscalar(mean):
                mshape = [1] * array.ndim
                mshape[0] = len(array) if len(self.mean) == 1 else len(self.mean)
                mean = np.array(self.mean, dtype=array.dtype).reshape(*mshape)
            if not np.isscalar(std):
                rshape = [1] * array.ndim
                rshape[0] = len(array) if len(self.std) == 1 else len(self.std)
                std = np.array(self.std, dtype=array.dtype).reshape(*rshape)
            return (array - mean) / std


    class Flatten:
        """Flatten a NumPy array.
        """
        def __call__(self, array):
            return array.flatten()


    class AsType:
        def __init__(self, dtype=np.float32):
            self.dtype = dtype

        def __call__(self, array):
            return array.astype(self.dtype)


    ToFloat = AsType


    class ToInt(AsType):
        def __init__(self, dtype=np.int32):
            self.dtype = dtype

class S_datasets:
    # =============================================================================
    # Toy datasets
    # =============================================================================
    def get_spiral(train=True):
        seed = 1984 if train else 2020
        np.random.seed(seed=seed)

        num_data, num_class, input_dim = 100, 3, 2
        data_size = num_class * num_data
        x = np.zeros((data_size, input_dim), dtype=np.float32)
        t = np.zeros(data_size, dtype=np.int32)

        for j in range(num_class):
            for i in range(num_data):
                rate = i / num_data
                radius = 1.0 * rate
                theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
                ix = num_data * j + i
                x[ix] = np.array([radius * np.sin(theta),
                                radius * np.cos(theta)]).flatten()
                t[ix] = j
        # Shuffle
        indices = np.random.permutation(num_data * num_class)
        x = x[indices]
        t = t[indices]
        return x, t

    class Spiral(datasets.Dataset):
        def prepare(self):
            self.data, self.label = s_datasets.get_spiral(self.train)


    # =============================================================================
    # MNIST-like dataset: MNIST / CIFAR /
    # =============================================================================
    class MNIST(datasets.Dataset):

        def __init__(self, train=True,
                    transform=transforms.Compose([transforms.Flatten(), transforms.ToFloat(),
                                        transforms.Normalize(0., 255.)]),
                    target_transform=None):
            super().__init__(train, transform, target_transform)

        def prepare(self):
            url = 'http://yann.lecun.com/exdb/mnist/'
            train_files = {'target': 'train-images-idx3-ubyte.gz',
                        'label': 'train-labels-idx1-ubyte.gz'}
            test_files = {'target': 't10k-images-idx3-ubyte.gz',
                        'label': 't10k-labels-idx1-ubyte.gz'}

            files = train_files if self.train else test_files
            data_path = S_utils.get_file(url + files['target'])
            label_path = S_utils.get_file(url + files['label'])

            self.data = self._load_data(data_path)
            self.label = self._load_label(label_path)

        def _load_label(self, filepath):
            with gzip.open(filepath, 'rb') as f:
                labels = np.frombuffer(f.read(), np.uint8, offset=8)
            return labels

        def _load_data(self, filepath):
            with gzip.open(filepath, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)
            data = data.reshape(-1, 1, 28, 28)
            return data

        def show(self, row=10, col=10):
            H, W = 28, 28
            img = np.zeros((H * row, W * col))
            for r in range(row):
                for c in range(col):
                    img[r * H:(r + 1) * H, c * W:(c + 1) * W] = self.data[
                        np.random.randint(0, len(self.data) - 1)].reshape(H, W)
            plt.imshow(img, cmap='gray', interpolation='nearest')
            plt.axis('off')
            plt.show()

        @staticmethod
        def labels():
            return {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}

    class CIFAR10(datasets.Dataset):

        def __init__(self, train=True,
                    transform=transforms.Compose([transforms.ToFloat(), transforms.Normalize(mean=0.5, std=0.5)]),
                    target_transform=None):
            super().__init__(train, transform, target_transform)

        def prepare(self):
            url='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
            self.data, self.label = S_datasets.load_cache_npz(url, self.train)
            if self.data is not None:
                return
            filepath = S_utils.get_file(url)
            if self.train:
                self.data = np.empty((50000, 3 * 32 * 32))
                self.label = np.empty((50000), dtype=np.int32)
                for i in range(5):
                    self.data[i * 10000:(i + 1) * 10000] = self._load_data(
                        filepath, i + 1, 'train')
                    self.label[i * 10000:(i + 1) * 10000] = self._load_label(
                        filepath, i + 1, 'train')
            else:
                self.data = self._load_data(filepath, 5, 'test')
                self.label = self._load_label(filepath, 5, 'test')
            self.data = self.data.reshape(-1, 3, 32, 32)
            S_datasets.save_cache_npz(self.data, self.label, url, self.train)


        def _load_data(self, filename, idx, data_type='train'):
            assert data_type in ['train', 'test']
            with tarfile.open(filename, 'r:gz') as file:
                for item in file.getmembers():
                    if ('data_batch_{}'.format(idx) in item.name and data_type == 'train') or ('test_batch' in item.name and data_type == 'test'):
                        data_dict = pickle.load(file.extractfile(item), encoding='bytes')
                        data = data_dict[b'data']
                        return data

        def _load_label(self, filename, idx, data_type='train'):
            assert data_type in ['train', 'test']
            with tarfile.open(filename, 'r:gz') as file:
                for item in file.getmembers():
                    if ('data_batch_{}'.format(idx) in item.name and data_type == 'train') or ('test_batch' in item.name and data_type == 'test'):
                        data_dict = pickle.load(file.extractfile(item), encoding='bytes')
                        return np.array(data_dict[b'labels'])

        def show(self, row=10, col=10):
            H, W = 32, 32
            img = np.zeros((H*row, W*col, 3))
            for r in range(row):
                for c in range(col):
                    img[r*H:(r+1)*H, c*W:(c+1)*W] = self.data[np.random.randint(0, len(self.data)-1)].reshape(3,H,W).transpose(1,2,0)/255
            plt.imshow(img, interpolation='nearest')
            plt.axis('off')
            plt.show()

        @staticmethod
        def labels():
            return {0: 'ariplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

    class CIFAR100(CIFAR10):

        def __init__(self, train=True,
                    transform=transforms.Compose([transforms.ToFloat(), transforms.Normalize(mean=0.5, std=0.5)]),
                    target_transform=None,
                    label_type='fine'):
            assert label_type in ['fine', 'coarse']
            self.label_type = label_type
            super().__init__(train, transform, target_transform)

        def prepare(self):
            url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
            self.data, self.label = S_datasets.load_cache_npz(url, self.train)
            if self.data is not None:
                return

            filepath = S_utils.get_file(url)
            if self.train:
                self.data = self._load_data(filepath, 'train')
                self.label = self._load_label(filepath, 'train')
            else:
                self.data = self._load_data(filepath, 'test')
                self.label = self._load_label(filepath, 'test')
            self.data = self.data.reshape(-1, 3, 32, 32)
            S_datasets.save_cache_npz(self.data, self.label, url, self.train)

        def _load_data(self, filename, data_type='train'):
            with tarfile.open(filename, 'r:gz') as file:
                for item in file.getmembers():
                    if data_type in item.name:
                        data_dict = pickle.load(file.extractfile(item), encoding='bytes')
                        data = data_dict[b'data']
                        return data

        def _load_label(self, filename, data_type='train'):
            assert data_type in ['train', 'test']
            with tarfile.open(filename, 'r:gz') as file:
                for item in file.getmembers():
                    if data_type in item.name:
                        data_dict = pickle.load(file.extractfile(item), encoding='bytes')
                        if self.label_type == 'fine':
                            return np.array(data_dict[b'fine_labels'])
                        elif self.label_type == 'coarse':
                            return np.array(data_dict[b'coarse_labels'])

        @staticmethod
        def labels(label_type='fine'):
            coarse_labels = dict(enumerate(['aquatic mammals','fish','flowers','food containers','fruit and vegetables','household electrical device','household furniture','insects','large carnivores','large man-made outdoor things','large natural outdoor scenes','large omnivores and herbivores','medium-sized mammals','non-insect invertebrates','people','reptiles','small mammals','trees','vehicles 1','vehicles 2']))
            fine_labels = []
            return fine_labels if label_type is 'fine' else coarse_labels

    # =============================================================================
    # Big datasets
    # =============================================================================
    class ImageNet(datasets.Dataset):

        def __init__(self):
            NotImplemented

        @staticmethod
        def labels():
            url = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt'
            path = S_utils.get_file(url)
            with open(path, 'r') as f:
                labels = eval(f.read())
            return labels

    # =============================================================================
    # Sequential datasets: SinCurve, Shapekspare
    # =============================================================================
    class SinCurve(datasets.Dataset):

        def prepare(self):
            num_data = 1000
            dtype = np.float64

            x = np.linspace(0, 2 * np.pi, num_data)
            noise_range = (-0.05, 0.05)
            noise = np.random.uniform(noise_range[0], noise_range[1], size=x.shape)
            if self.train:
                y = np.sin(x) + noise
            else:
                y = np.cos(x)
            y = y.astype(dtype)
            self.data = y[:-1][:, np.newaxis]
            self.label = y[1:][:, np.newaxis]

    class Shakespear(datasets.Dataset):

        def prepare(self):
            url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
            file_name = 'shakespear.txt'
            path = S_utils.get_file(url, file_name)
            with open(path, 'r') as f:
                data = f.read()
            chars = list(data)

            char_to_id = {}
            id_to_char = {}
            for word in data:
                if word not in char_to_id:
                    new_id = len(char_to_id)
                    char_to_id[word] = new_id
                    id_to_char[new_id] = word

            indices = np.array([char_to_id[c] for c in chars])
            self.data = indices[:-1]
            self.label = indices[1:]
            self.char_to_id = char_to_id
            self.id_to_char = id_to_char

    # =============================================================================
    # Utils
    # =============================================================================
    def load_cache_npz(filename, train=False):
        filename = filename[filename.rfind('/') + 1:]
        prefix = '.train.npz' if train else '.test.npz'
        filepath = os.path.join(S_utils.cache_dir, filename + prefix)
        if not os.path.exists(filepath):
            return None, None

        loaded = np.load(filepath)
        return loaded['data'], loaded['label']

    def save_cache_npz(data, label, filename, train=False):
        filename = filename[filename.rfind('/') + 1:]
        prefix = '.train.npz' if train else '.test.npz'
        filepath = os.path.join(S_utils.cache_dir, filename + prefix)

        if os.path.exists(filepath):
            return

        print("Saving: " + filename + prefix)
        try:
            np.savez_compressed(filepath, data=data, label=label)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            raise
        print(" Done")
        return filepath

class S_Models:

    # =============================================================================
    # VGG
    # =============================================================================
    class VGG16(Models.Model):
        WEIGHTS_PATH = 'https://github.com/koki0702/dezero-models/releases/download/v0.1/vgg16.npz'

        def __init__(self, pretrained=False):
            super().__init__()
            self.conv1_1 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
            self.conv1_2 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
            self.conv2_1 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
            self.conv2_2 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
            self.conv3_1 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
            self.conv3_2 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
            self.conv3_3 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
            self.conv4_1 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
            self.conv4_2 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
            self.conv4_3 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
            self.conv5_1 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
            self.conv5_2 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
            self.conv5_3 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
            self.fc6 = L.Linear(4096)
            self.fc7 = L.Linear(4096)
            self.fc8 = L.Linear(1000)

            if pretrained:
                weights_path = utils.get_file(VGG16.WEIGHTS_PATH)
                self.load_weights(weights_path)

        def forward(self, x):
            x = F.relu(self.conv1_1(x))
            x = F.relu(self.conv1_2(x))
            x = F.pooling(x, 2, 2)
            x = F.relu(self.conv2_1(x))
            x = F.relu(self.conv2_2(x))
            x = F.pooling(x, 2, 2)
            x = F.relu(self.conv3_1(x))
            x = F.relu(self.conv3_2(x))
            x = F.relu(self.conv3_3(x))
            x = F.pooling(x, 2, 2)
            x = F.relu(self.conv4_1(x))
            x = F.relu(self.conv4_2(x))
            x = F.relu(self.conv4_3(x))
            x = F.pooling(x, 2, 2)
            x = F.relu(self.conv5_1(x))
            x = F.relu(self.conv5_2(x))
            x = F.relu(self.conv5_3(x))
            x = F.pooling(x, 2, 2)
            x = F.reshape(x, (x.shape[0], -1))
            x = F.dropout(F.relu(self.fc6(x)))
            x = F.dropout(F.relu(self.fc7(x)))
            x = self.fc8(x)
            return x

        @staticmethod
        def preprocess(image, size=(224, 224), dtype=np.float32):
            image = image.convert('RGB')
            if size:
                image = image.resize(size)
            image = np.asarray(image, dtype=dtype)
            image = image[:, :, ::-1]
            image -= np.array([103.939, 116.779, 123.68], dtype=dtype)
            image = image.transpose((2, 0, 1))
            return image

    # =============================================================================
    # ResNet
    # =============================================================================
    class ResNet(Models.Model):
        WEIGHTS_PATH = 'https://github.com/koki0702/dezero-models/releases/download/v0.1/resnet{}.npz'

        def __init__(self, n_layers=152, pretrained=False):
            super().__init__()

            if n_layers == 50:
                block = [3, 4, 6, 3]
            elif n_layers == 101:
                block = [3, 4, 23, 3]
            elif n_layers == 152:
                block = [3, 8, 36, 3]
            else:
                raise ValueError('The n_layers argument should be either 50, 101,'
                                ' or 152, but {} was given.'.format(n_layers))

            self.conv1 = L.Conv2d(64, 7, 2, 3)
            self.bn1 = L.BatchNorm()
            self.res2 = S_Models.BuildingBlock(block[0], 64, 64, 256, 1)
            self.res3 = S_Models.BuildingBlock(block[1], 256, 128, 512, 2)
            self.res4 = S_Models.BuildingBlock(block[2], 512, 256, 1024, 2)
            self.res5 = S_Models.BuildingBlock(block[3], 1024, 512, 2048, 2)
            self.fc6 = L.Linear(1000)

            if pretrained:
                weights_path = S_utils.get_file(ResNet.WEIGHTS_PATH.format(n_layers))
                self.load_weights(weights_path)

        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.pooling(x, kernel_size=3, stride=2)
            x = self.res2(x)
            x = self.res3(x)
            x = self.res4(x)
            x = self.res5(x)
            x = S_Models._global_average_pooling_2d(x)
            x = self.fc6(x)
            return x

    class ResNet152(ResNet):
        def __init__(self, pretrained=False):
            super().__init__(152, pretrained)

    class ResNet101(ResNet):
        def __init__(self, pretrained=False):
            super().__init__(101, pretrained)

    class ResNet50(ResNet):
        def __init__(self, pretrained=False):
            super().__init__(50, pretrained)

    def _global_average_pooling_2d(x):
        N, C, H, W = x.shape
        h = F.average_pooling(x, (H, W), stride=1)
        h = F.reshape(h, (N, C))
        return h

    class BuildingBlock(L.Layer):
        def __init__(self, n_layers=None, in_channels=None, mid_channels=None,
                    out_channels=None, stride=None, downsample_fb=None):
            super().__init__()

            self.a = S_Models.BottleneckA(in_channels, mid_channels, out_channels, stride,
                                downsample_fb)
            self._forward = ['a']
            for i in range(n_layers - 1):
                name = 'b{}'.format(i+1)
                bottleneck = S_Models.BottleneckB(out_channels, mid_channels)
                setattr(self, name, bottleneck)
                self._forward.append(name)

        def forward(self, x):
            for name in self._forward:
                l = getattr(self, name)
                x = l(x)
            return x

    class BottleneckA(L.Layer):
        def __init__(self, in_channels, mid_channels, out_channels,
                    stride=2, downsample_fb=False):
            super().__init__()
            # In the original MSRA ResNet, stride=2 is on 1x1 convolution.
            # In Facebook ResNet, stride=2 is on 3x3 convolution.
            stride_1x1, stride_3x3 = (1, stride) if downsample_fb else (stride, 1)
        
            self.conv1 = L.Conv2d(mid_channels, 1, stride_1x1, 0,
                                nobias=True)
            self.bn1 = L.BatchNorm()
            self.conv2 = L.Conv2d(mid_channels, 3, stride_3x3, 1,
                                nobias=True)
            self.bn2 = L.BatchNorm()
            self.conv3 = L.Conv2d(out_channels, 1, 1, 0, nobias=True)
            self.bn3 = L.BatchNorm()
            self.conv4 = L.Conv2d(out_channels, 1, stride, 0,
                                nobias=True)
            self.bn4 = L.BatchNorm()

        def forward(self, x):
            h1 = F.relu(self.bn1(self.conv1(x)))
            h1 = F.relu(self.bn2(self.conv2(h1)))
            h1 = self.bn3(self.conv3(h1))
            h2 = self.bn4(self.conv4(x))
            return F.relu(h1 + h2)

    class BottleneckB(L.Layer):
        def __init__(self, in_channels, mid_channels):
            super().__init__()
            
            self.conv1 = L.Conv2d(mid_channels, 1, 1, 0, nobias=True)
            self.bn1 = L.BatchNorm()
            self.conv2 = L.Conv2d(mid_channels, 3, 1, 1, nobias=True)
            self.bn2 = L.BatchNorm()
            self.conv3 = L.Conv2d(in_channels, 1, 1, 0, nobias=True)
            self.bn3 = L.BatchNorm()

        def forward(self, x):
            h = F.relu(self.bn1(self.conv1(x)))
            h = F.relu(self.bn2(self.conv2(h)))
            h = self.bn3(self.conv3(h))
            return F.relu(h + x)
    # =============================================================================
    # SqueezeNet
    # =============================================================================
    class SqueezeNet(Models.Model):
        def __init__(self, pretrained=False):
            pass

        def forward(self, x):
            pass


