python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18 \
     --save-name pgd-l-infinite-20_8-untargeted

python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --targeted \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18 \
     --save-name pgd-l-infinite-20_8-targeted

python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --targeted \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-adv-0.01/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-adv-0.01 \
     --save-name pgd-l-infinite-20_8-targeted



python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-adv-0.01/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-adv-0.01 \
     --save-name pgd-l-infinite-20_8-untargeted


     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-adv-0.03/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-adv-0.03 \
     --save-name pgd-l-infinite-20_8-untargeted


     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --targeted \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-adv-0.03/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-adv-0.03 \
     --save-name pgd-l-infinite-20_8-targeted


     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-adv-0.05/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-adv-0.05 \
     --save-name pgd-l-infinite-20_8-untargeted


     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --targeted \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-adv-0.05/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-adv-0.05 \
     --save-name pgd-l-infinite-20_8-targeted


     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-adv-0.1/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-adv-0.1 \
     --save-name pgd-l-infinite-20_8-untargeted


    python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --targeted \
     --adv-type pgd \
     --resume \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-adv-0.1/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-adv-0.1 \
     --save-name pgd-l-infinite-20_8-targeted



     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-adv-0.2/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-adv-0.2 \
     --save-name pgd-l-infinite-20_8-untargeted


     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --targeted \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-adv-0.2/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-adv-0.2 \
     --save-name pgd-l-infinite-20_8-targeted


     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-adv-0.5/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-adv-0.5 \
     --save-name pgd-l-infinite-20_8-untargeted


     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --targeted \
     --adv-type pgd \
     --resume \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-adv-0.5/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-adv-0.5 \
     --save-name pgd-l-infinite-20_8-targeted

     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-adv-1/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-adv-1 \
     --save-name pgd-l-infinite-20_8-untargeted




     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --targeted \
     --adv-type pgd \
     --resume \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-adv-1/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-adv-1 \
     --save-name pgd-l-infinite-20_8-targeted




python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --targeted \
     --resume \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-jr-1/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-jr-1 \
     --save-name pgd-l-infinite-20_8-targeted

python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-jr-1/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-jr-1 \
     --save-name pgd-l-infinite-20_8-untargeted

python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-jr-0.5/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-jr-0.5 \
     --save-name pgd-l-infinite-20_8-untargeted

python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-jr-0.5/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-jr-0.5 \
     --save-name pgd-l-infinite-20_8-targeted


python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-ig-0.5/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-ig-0.5 \
     --save-name pgd-l-infinite-20_8-targeted

python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-ig-0.5/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-ig-0.5 \
     --save-name pgd-l-infinite-20_8-untargeted


python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-ig-1/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-ig-1 \
     --save-name pgd-l-infinite-20_8-untargeted

python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-ig-1/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-ig-1 \
     --save-name pgd-l-infinite-20_8-targeted

python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-jr-1.5/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-jr-1.5 \
     --save-name pgd-l-infinite-20_8-targeted


     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-jr-2/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-jr-2 \
     --save-name pgd-l-infinite-20_8-targeted


     python forge_attack.py \
     --on-test-set \
     --arch PreActResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/preactresnet18-mixup/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/preactresnet18-mixup \
     --save-name pgd-l-infinite-20_8-targeted

     python forge_attack.py \
     --on-test-set \
     --arch PreActResNet18 \
     --dataset cifar10 \
     --batch-size 1000 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/preactresnet18/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/preactresnet18 \
     --save-name pgd-l-infinite-20_8-targeted



     python forge_attack.py \
     --on-test-set \
     --arch PreActResNet18 \
     --dataset cifar10 \
     --batch-size 1000 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/preactresnet18-jr-1-mine-sam-0.1/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/preactresnet18-jr-1-mine-sam-0.1 \
     --save-name pgd-l-infinite-20_8-targeted


python forge_attack.py \
     --on-test-set \
     --arch VGG16 \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-jr-0.00001/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-jr-0.00001 \
     --save-name pgd-l-infinite-20_8-targeted


python forge_attack.py \
     --on-test-set \
     --arch VGG16 \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/vgg16/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/vgg16 \
     --save-name pgd-l-infinite-20_8-targeted

python forge_attack.py \
     --on-test-set \
     --arch VGG16 \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/vgg16-jr-1-mine/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/vgg16-jr-1-mine \
     --save-name pgd-l-infinite-20_8-targeted

python forge_attack.py \
     --on-test-set \
     --arch LeNet \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/lenet-jr-1-mine/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/lenet-jr-1-mine \
     --save-name pgd-l-infinite-20_8-targeted

python forge_attack.py \
     --on-test-set \
     --arch LeNet \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/lenet/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/lenet \
     --save-name pgd-l-infinite-20_8-targeted

     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-jr-0.1/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-jr-0.1 \
     --save-name pgd-l-infinite-20_8-targeted


     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-jr-0.01/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-jr-0.01 \
     --save-name pgd-l-infinite-20_8-targeted


     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-jr-0.001/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-jr-0.001 \
     --save-name pgd-l-infinite-20_8-targeted



     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-jr-0.0001/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-jr-0.0001 \
     --save-name pgd-l-infinite-20_8-targeted


     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-jr-0.00001/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-jr-0.00001 \
     --save-name pgd-l-infinite-20_8-targeted


     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-jr-0.000001/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-jr-0.000001 \
     --save-name pgd-l-infinite-20_8-targeted


     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-jr-1.25/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-jr-1.25 \
     --save-name pgd-l-infinite-20_8-targeted


     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-jr-0.0005/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-jr-0.0005 \
     --save-name pgd-l-infinite-20_8-targeted


     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-model-0.001/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-model-0.001 \
     --save-name pgd-l-infinite-20_8-targeted


     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-model-0.01/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-model-0.01 \
     --save-name pgd-l-infinite-20_8-targeted


     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-model-1/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-model-1 \
     --save-name pgd-l-infinite-20_8-targeted


     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-jr-1.75-mine/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-jr-1.75-mine \
     --save-name pgd-l-infinite-20_8-targeted

     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-jr-1.25-mine/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-jr-1.25-mine \
     --save-name pgd-l-infinite-20_8-targeted


     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-logits-model-0.01/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-logits-model-0.01 \
     --save-name pgd-l-infinite-20_8-targeted

     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-logits-model-0.1/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-logits-model-0.1 \
     --save-name pgd-l-infinite-20_8-targeted



     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-logits-model-0.001/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-logits-model-0.001 \
     --save-name pgd-l-infinite-20_8-targeted



     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-model-0.1/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-model-0.1 \
     --save-name pgd-l-infinite-20_8-targeted

     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-model-0.2/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-model-0.2 \
     --save-name pgd-l-infinite-20_8-targeted

     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-model-0.5/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-model-0.5 \
     --save-name pgd-l-infinite-20_8-targeted


     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-model-0.001/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-model-0.001 \
     --save-name pgd-l-infinite-20_8-targeted

     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-model-0.0001/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-model-0.0001 \
     --save-name pgd-l-infinite-20_8-targeted


     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-mixup-0.1/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-mixup-0.1 \
     --save-name pgd-l-infinite-20_8-targeted

     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-mixup-0.2/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-mixup-0.2 \
     --save-name pgd-l-infinite-20_8-targeted


     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-mixup-0.3/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-mixup-0.3 \
     --save-name pgd-l-infinite-20_8-targeted

     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-mixup-0.5/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-mixup-0.5 \
     --save-name pgd-l-infinite-20_8-targeted


     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-sam-1/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-sam-1 \
     --save-name pgd-l-infinite-20_8-targeted


     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-sam-1/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-sam-1 \
     --save-name pgd-l-infinite-20_8-targeted

python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 1 \
     --pgd-steps 20  \
     --pgd-step-size  0.125 \
     --adv-type pgd \
     --pgd-norm-type l2\
     --targeted \
     --resume \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18\
     --save-name pgd-l2-20_1-targeted

python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 1 \
     --pgd-steps 20  \
     --pgd-step-size  0.125 \
     --adv-type pgd \
     --pgd-norm-type l2\
     --targeted \
     --resume \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-adv-0.01/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-adv-0.01\
     --save-name pgd-l2-20_1-targeted

python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 1 \
     --pgd-steps 20  \
     --pgd-step-size  0.125 \
     --adv-type pgd \
     --pgd-norm-type l2\
     --targeted \
     --resume \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-adv-0.03/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-adv-0.03\
     --save-name pgd-l2-20_1-targeted

python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 1 \
     --pgd-steps 20  \
     --pgd-step-size  0.125 \
     --adv-type pgd \
     --pgd-norm-type l2\
     --targeted \
     --resume \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-adv-0.05/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-adv-0.05\
     --save-name pgd-l2-20_1-targeted


python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 1 \
     --pgd-steps 20  \
     --pgd-step-size  0.125 \
     --adv-type pgd \
     --pgd-norm-type l2\
     --targeted \
     --resume \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-adv-0.1/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-adv-0.1\
     --save-name pgd-l2-20_1-targeted



python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 1 \
     --pgd-steps 20  \
     --pgd-step-size  0.125 \
     --adv-type pgd \
     --pgd-norm-type l2\
     --targeted \
     --resume \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-adv-0.3/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-adv-0.3\
     --save-name pgd-l2-20_1-targeted


python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 1 \
     --pgd-steps 20  \
     --pgd-step-size  0.125 \
     --adv-type pgd \
     --pgd-norm-type l2\
     --targeted \
     --resume \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-adv-0.5/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-adv-0.5\
     --save-name pgd-l2-20_1-targeted

python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 1 \
     --pgd-steps 20  \
     --pgd-step-size  0.125 \
     --adv-type pgd \
     --pgd-norm-type l2\
     --targeted \
     --resume \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-adv-1/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-adv-1\
     --save-name pgd-l2-20_1-targeted


python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 0.5 \
     --pgd-steps 20  \
     --pgd-step-size  0.0625 \
     --adv-type pgd \
     --pgd-norm-type l2\
     --targeted \
     --resume \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-sam-0.1/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-sam-0.1\
     --save-name pgd-l2-20_0.5-targeted

python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 0.5 \
     --pgd-steps 20  \
     --pgd-step-size  0.0625 \
     --adv-type pgd \
     --pgd-norm-type l2\
     --targeted \
     --resume \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-sam-0.3/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-sam-0.3\
     --save-name pgd-l2-20_0.5-targeted

python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 0.5 \
     --pgd-steps 20  \
     --pgd-step-size  0.0625 \
     --adv-type pgd \
     --pgd-norm-type l2\
     --targeted \
     --resume \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-sam-0.05/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-sam-0.05\
     --save-name pgd-l2-20_0.5-targeted



python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 0.5 \
     --pgd-steps 20  \
     --pgd-step-size  0.0625 \
     --adv-type pgd \
     --pgd-norm-type l2\
     --targeted \
     --resume \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-sam-0.2/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-sam-0.2\
     --save-name pgd-l2-20_0.5-targeted

python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 1 \
     --pgd-steps 20  \
     --pgd-step-size  0.125 \
     --adv-type pgd \
     --pgd-norm-type l2\
     --targeted \
     --resume \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-sam-0.2/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-sam-0.2\
     --save-name pgd-l2-20_1-targeted


python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 0.5 \
     --pgd-steps 20  \
     --pgd-step-size  0.0625 \
     --adv-type pgd \
     --pgd-norm-type l2\
     --targeted \
     --resume \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-sam-0.4/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-sam-0.4\
     --save-name pgd-l2-20_0.5-targeted

python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 1 \
     --pgd-steps 20  \
     --pgd-step-size  0.125 \
     --adv-type pgd \
     --pgd-norm-type l2\
     --targeted \
     --resume \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-sam-0.4/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-sam-0.4\
     --save-name pgd-l2-20_1-targeted


     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-cutmix-0.1/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-cutmix-0.1 \
     --save-name pgd-l-infinite-20_8-targeted


     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-cutmix-0.3/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-cutmix-0.3 \
     --save-name pgd-l-infinite-20_8-targeted


     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-cutmix-0.5/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-cutmix-0.5 \
     --save-name pgd-l-infinite-20_8-targeted


     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 250 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-cutmix-0.7/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-cutmix-0.7 \
     --save-name pgd-l-infinite-20_8-targeted


     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-samemixup-0.7/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-samemixup-0.7 \
     --save-name pgd-l-infinite-20_8-targeted

     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-samemixup-0.1/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-samemixup-0.1 \
     --save-name pgd-l-infinite-20_8-targeted

     python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-samemixup-0.3/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-samemixup-0.3 \
     --save-name pgd-l-infinite-20_8-targeted


          python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-samemixup-0.5/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-samemixup-0.5 \
     --save-name pgd-l-infinite-20_8-targeted



python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-TRADE-0.03/best-robust-eval-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-TRADE-0.03 \
     --save-name pgd-l-infinite-20_8-targeted-best-robust



python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-TRADE-0.03/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-TRADE-0.03 \
     --save-name pgd-l-infinite-20_8-targeted

python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-TRADE-0.05/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-TRADE-0.05 \
     --save-name pgd-l-infinite-20_8-targeted

python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-TRADE-0.1/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-TRADE-0.1 \
     --save-name pgd-l-infinite-20_8-targeted

python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-TRADE-0.2/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-TRADE-0.2 \
     --save-name pgd-l-infinite-20_8-targeted

python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-TRADE-0.5/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-TRADE-0.5 \
     --save-name pgd-l-infinite-20_8-targeted


python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-TRADE-0.01/best-robust-eval-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-TRADE-0.01 \
     --save-name pgd-l-infinite-20_8-targeted-best-robust


python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-TRADE-0.03/best-robust-eval-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-TRADE-0.03 \
     --save-name pgd-l-infinite-20_8-targeted-best-robust

python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-TRADE-0.05/best-robust-eval-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-TRADE-0.05 \
     --save-name pgd-l-infinite-20_8-targeted-best-robust


python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-TRADE-0.1/best-robust-eval-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-TRADE-0.1 \
     --save-name pgd-l-infinite-20_8-targeted-best-robust


python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-TRADE-0.2/best-robust-eval-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-TRADE-0.2 \
     --save-name pgd-l-infinite-20_8-targeted-best-robust

python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-TRADE-0.5/best-robust-eval-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-TRADE-0.5 \
     --save-name pgd-l-infinite-20_8-targeted-best-robust

python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-jr-sam/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-jr-sam \
     --save-name pgd-l-infinite-20_8-targeted

python forge_attack.py \
     --on-test-set \
     --arch ResNet18 \
     --dataset cifar10 \
     --batch-size 500 \
     --pgd-radius 8 \
     --pgd-steps 20  \
     --pgd-step-size  2 \
     --adv-type pgd \
     --resume \
     --targeted \
     --resume-path ./CIFAR_Train/cifar10-models/resnet18-jr-sam-1-2/model-fin-model.pkl\
     --data-dir ./CIFAR_Train/data \
     --save-dir ./CIFAR_Train/cifar10-models/resnet18-jr-sam-1-2 \
     --save-name pgd-l-infinite-20_8-targeted
