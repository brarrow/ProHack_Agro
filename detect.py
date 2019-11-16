import image
from status import Status
from persons import Person

status_ = Status()


def detecting(resized_im, seg_map, N=False,  H=False, P=False, G=False):
    img_with_mask = image.bitwise_images(resized_im, seg_map)
    res_img = img_with_mask.copy()
    # here need to split persons on image. All bottom methods for one person
    persons_masks = [img_with_mask]
    status_.persons = [Person(mask) for mask in persons_masks]

    for person in status_.persons:
        if N:
            pass
        if H:
            detect_helmets(person)
        if P:
            pass
        if G:
            pass
    # status check for all persons
    status_.check_status()
    status_.clear()
    return res_img


def detect_helmets(person):

    return person.H


def cut_head(person):

    pass