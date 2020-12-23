
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



if __name__ == "__main__":
    import torchvision.models as models
    model = models.alexnet(pretrained=True)
    from my_lib.path.path_manager import OutputPathManager
    path_manager = OutputPathManager()
    pdf_file = path_manager.create_output_file("temp", "model.pdf")
    draw_nn_model(model, pdf_file_name=pdf_file)