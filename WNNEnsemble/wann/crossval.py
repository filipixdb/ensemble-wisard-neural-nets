import itertools as it
import multiprocessing as mp

import ensemble.bootstrap as boot

def make_folds(data, number_folds=10):
    """Returns data partitioned into the given number of folds."""

    #rnd.shuffle(data)
    return [data[i::number_folds] for i in xrange(number_folds)]


def cross_validate(
        folds, classifier, ans_method='counts', bagging=None, percent=0.5, encoder=None):
    """Cross-validate the classifier using data in folds.

    The procedure has as many turns as the number of data folds. On
    i-th turn all folds except the i-th are used to train the
    classifier. Then the i-th fold is used to test the classifier.

    Parameters:
        folds: data, divided in folds. See function make_folds.
        classifier: a callable which returns a classifier object.
        ans_method: the name of the classifier method which provides
            answers to classification queries.
        encoder: a callable which receives an iterator to some
            training data as argument and returns a data encoder.

    Yields:
        For each fold it is yielded, in this order:
            1. A tuple (classifier_obj,
                        encoder_obj if encoder is not None else None)
            2. For each observation in the test fold, a tuple
                (observation, classifier_answers, real_class)
            3. A marker of the end of the current test fold: None

        A suggestion for the loop structure to get the results of this
        function is the following:

            cv_gen = cross_validate(folds, classifier)

            for classifier_obj, encoder_obj in cv_gen:
                for fold_item in cv_gen:
                    if fold_item is None:
                        break  # end-of-fold marker

                    observation, answers, real_class = fold_item
                    # do stuff
    """

    cv_args = (folds, classifier, ans_method, bagging, percent, encoder)

# itera em todos os foldes
#retorna varios (classificador,(observacao, classificacao, classe_real)) 
    for i in xrange(len(folds)):
        for result in _run_turn((i, cv_args)):
            yield result

        yield None  # end-of-fold token


def _run_turn(args):
    tf, (folds, classifier, ans_method, bagging, percent, encoder) = args

    if encoder:
        # iterator over all observations except those of fold tf
        train_folds = (
            fold for fi, fold in enumerate(folds) if fi != tf)
        encoder_obj = encoder(it.chain(*train_folds))

    else:
        encoder_obj = lambda x: x

    # iterate over all observations except those of fold tf
    train_folds = (fold for fi, fold in enumerate(folds) if fi != tf)
    classifier_obj = classifier()

### AQUI EU TENHO QUE FAZER ELE USAR ALEATORIAMENTE X% DO DATASET DE TREINO

    treino = []
    
    if bagging:
        if bagging == 'no_repetition':
            treino = boot.bootstrap(train_folds, percent)
        if bagging == 'repetition':
            treino = boot.bootstrap_wr(train_folds, percent)
    else:
        treino = [z for z in it.chain(*train_folds)]


# itera em todas as instancias de treino

#    for ax in it.chain(*train_folds):
    for ax in treino:
        classifier_obj.record(*encoder_obj(ax))

    # yield the trained classifier and the encoder of run tf
    yield classifier_obj, encoder_obj if encoder else None

    for observation, class_ in it.imap(encoder_obj, folds[tf]):
        yield (observation,
               # pede a funcao de resposta do classificador, e chama ela passando a observation
               getattr(classifier_obj, ans_method)(observation),
               class_)


def cross_validate_multiproc(
        folds, classifier, ans_method='counts', encoder=None):
    """A multiprocess version of cross_validate function."""

    queue = mp.Queue()
    cv_args = (folds, classifier, ans_method, encoder)

    for i in xrange(len(folds)):
        p = mp.Process(
            target=_run_turn_multiproc, args=(queue, (i, cv_args)))
        p.start()

    for _ in xrange(len(folds)):
        for res in queue.get():
            yield res

        yield None  # end-of-fold token


def _run_turn_multiproc(queue, cv_args):
    queue.put(list(_run_turn(cv_args)))

