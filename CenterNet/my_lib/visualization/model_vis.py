
def draw_nn_model(model, pdf_file_name, input_shape=(1, 3, 244, 244)):
    """
    This function is used to draw a pytorch model in a PDF and PNG file

    :params model a torch.nn.Module
    :params pdf_file_name the name of the architecture PDF file
    :params input_shape (batch_num, band_num, row, col)

    """
    import torch
    from torchviz import make_dot
    x = torch.randn(*input_shape)
    y = model(x)
    show_tensor = None
    if isinstance(y, torch.Tensor):
        show_tensor = y
    elif isinstance(y, list):
        show_tensor = tuple(y[0].values())

    make_dot(show_tensor).render(pdf_file_name, view=False)
    pdf_file_name.rename(pdf_file_name.with_suffix('.png'))
    make_dot(show_tensor).render(pdf_file_name, view=False, format="png")

def describe_nn_model(model, text_file_name, input_shape=(3, 244, 244)):
    """
    This function is to write the model description in a text file.

    :params text_file_name text file name
    :params input_shape input image size (bands, row, col)

    """


    import torch
    from torchsummary import summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model2 = model.to(device)

    with open(text_file_name, "w") as fid:
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            summary(model2, input_size=input_shape)
        model_description = f.getvalue()
        fid.write(model_description)




if __name__ == "__main__":
    pass
    # model drawing
    # import torchvision.models as models
    # model = models.alexnet(pretrained=True)
    # from my_lib.path.path_manager import OutputPathManager
    # path_manager = OutputPathManager()
    # pdf_file = path_manager.create_output_file("temp", "model.pdf")
    # draw_nn_model(model, pdf_file_name=pdf_file)
    #
    # # model describiton
    # text_file = path_manager.create_output_file("temp", "model.txt")
    # describe_nn_model(model, text_file)