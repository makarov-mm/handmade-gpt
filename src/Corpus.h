// Built-in training corpus: original text, deliberately plain and
// repetitive so a character model picks up the patterns quickly.
#pragma once

#include <string_view>

namespace gpt {

inline constexpr std::string_view kCorpus =
    "the river rises in the hills and the river runs down to the sea. the rain "
    "falls on the slopes and the water gathers in the valley. the wind moves "
    "over the water and the waves follow the wind across the lake. the sun "
    "climbs in the morning and the sun settles behind the ridge in the evening. "
    "the light fades in the west and the stars come out over the fields. "
    "the birds wake before the light and the birds sing until the heat of the "
    "day. the leaves turn in the autumn and the leaves fall when the frost "
    "comes. the snow covers the ground in winter and the snow melts into the "
    "streams in spring. the seeds wait in the soil and the seeds open when the "
    "ground warms. the roots reach down for the water and the branches reach up "
    "for the light. the moon pulls the tide up the shore and the tide slides "
    "back down the sand. the fog settles in the low ground at night and the fog "
    "lifts when the morning warms the air. the path follows the stream through "
    "the woods and the path climbs out of the trees onto the open hill. the "
    "stone holds the heat of the day and the stone gives the heat back in the "
    "dark. the fire needs the air and the fire eats the wood down to the ash. "
    "the bread rises in the warmth and the bread browns in the heat of the "
    "oven. the letters make the words and the words make the lines and the "
    "lines fill the page. the reader follows the lines and the story unfolds "
    "one word at a time. the old man mends the net in the morning and the old "
    "man sails with the tide in the afternoon. the harvest comes at the end of "
    "the summer and the fields rest under the snow until the spring returns. ";

} // namespace gpt
