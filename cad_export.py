import ezdxf


def export_walls_to_dxf(walls, output_path):
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    doc.layers.new(name="WALL_AXIS", dxfattribs={"color": 3})

    for w in walls:
        if w.orientation == "horizontal":
            msp.add_line(
                (w.start, -w.coord),
                (w.end, -w.coord),
                dxfattribs={"layer": "WALL_AXIS"},
            )
        else:
            msp.add_line(
                (w.coord, -w.start),
                (w.coord, -w.end),
                dxfattribs={"layer": "WALL_AXIS"},
            )

    doc.saveas(output_path)
