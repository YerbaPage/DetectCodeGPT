import datetime
import os


def preprocess_and_save(args):
    START_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
    START_TIME = datetime.datetime.now().strftime('%H-%M-%S-%f')
    sampling_string = "top_k" if args.do_top_k else ("top_p" if args.do_top_p else "temp")
    output_subfolder = f"{args.output_name}/" if args.output_name else ""
    base_model_name = args.base_model_name.replace('/', '_')
    scoring_model_string = (f"-{args.scoring_model_name}" if args.scoring_model_name else "").replace('/', '_')
    SAVE_FOLDER = f"results/{output_subfolder}/{base_model_name}-{scoring_model_string}-{args.mask_filling_model_name}-{sampling_string}-{args.pct_words_masked}-{args.n_perturbation_rounds}-{args.n_samples}-{args.temperature}/{args.dataset}/{START_DATE}-{START_TIME}"
    


    cache_dir = args.cache_dir
    os.environ["XDG_CACHE_HOME"] = cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    print(f"Using cache dir {cache_dir}")
    return cache_dir, base_model_name, SAVE_FOLDER

    